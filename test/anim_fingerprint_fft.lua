--!strict
-- Shazam-style animation fingerprinting: convert joint motion to a 1D "audio-like"
-- signal, run STFT (FFT over windows), find peaks in time-frequency, build
-- constellation hashes. Compare by counting overlapping hashes.
--
-- Run with roblox_cli:
--   roblox-cli run --run <path>/anim_fingerprint_fft.lua --fs.readwrite <path> --load.asRobloxScript
--
-- Output: ff_fingerprints.csv (animId, clipId, duration, time_offset, hash)
--         One row per hash; match clips by counting shared hashes.

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

-- Config: use relative paths when run via roblox-cli --fs.readwrite <repo>
local CLIPS_DIR = "out/clips"
local OUTPUT_CSV = "ff_fingerprints.csv"
local FPS = 60
local NORMALIZE_DURATION = 2.0  -- seconds of clip to use (more = more FFT resolution)
local WINDOW_SIZE = 256         -- FFT window (power of 2)
local HOP_SIZE = 128
local PEAK_NEIGHBORHOOD = 3     -- 3x3 local max for peak finding
local FAN_VALUE = 5             -- pair each peak with up to 5 later peaks (Shazam "fan out")
local MAX_DT = 5                 -- max time bins between anchor and pair peak

local bit32 = bit32

-- ---------------------------------------------------------------------------
-- R15 character and animator (same as anim_fingerprint.lua)
-- ---------------------------------------------------------------------------
local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "AnimFingerprintR15"
	character.Parent = workspace
	if character.PrimaryPart then
		character:SetPrimaryPartCFrame(cframe)
	else
		local hrp = character:FindFirstChild("HumanoidRootPart")
		if hrp then hrp.CFrame = cframe end
	end
	return character
end

local character = spawnR15(CFrame.new(0, 0, 0))
local humanoid = character.Humanoid
local animator = humanoid.Animator
local hrp = character:FindFirstChild("HumanoidRootPart") :: BasePart

local R15_BONES = {
	"LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}

-- ---------------------------------------------------------------------------
-- Types
-- ---------------------------------------------------------------------------
type AnimTransform = { pos: Vector3, rot: CFrame }
type TrackFrames = { AnimTransform }
type ClipData = { [string]: TrackFrames }

-- ---------------------------------------------------------------------------
-- Math: quaternion, quat log (for motion signal)
-- ---------------------------------------------------------------------------
local function cframeToQuat(cf: CFrame): (number, number, number, number)
	local _, _, _, m00, m01, m02, m10, m11, m12, m20, m21, m22 = cf:GetComponents()
	local trace = m00 + m11 + m22
	local x, y, z, w
	if trace > 0 then
		local s = math.sqrt(trace + 1.0) * 2
		w = 0.25 * s
		x = (m21 - m12) / s
		y = (m02 - m20) / s
		z = (m10 - m01) / s
	elseif (m00 > m11) and (m00 > m22) then
		local s = math.sqrt(1.0 + m00 - m11 - m22) * 2
		w = (m21 - m12) / s
		x = 0.25 * s
		y = (m01 + m10) / s
		z = (m02 + m20) / s
	elseif m11 > m22 then
		local s = math.sqrt(1.0 + m11 - m00 - m22) * 2
		w = (m02 - m20) / s
		x = (m01 + m10) / s
		y = 0.25 * s
		z = (m12 + m21) / s
	else
		local s = math.sqrt(1.0 + m22 - m00 - m11) * 2
		w = (m10 - m01) / s
		x = (m02 + m20) / s
		y = (m12 + m21) / s
		z = 0.25 * s
	end
	local len = math.sqrt(x * x + y * y + z * z + w * w)
	return x / len, y / len, z / len, w / len
end

local function quatHemisphereAlign(ax, ay, az, aw, bx, by, bz, bw)
	local dot = ax * bx + ay * by + az * bz + aw * bw
	if dot < 0 then return -bx, -by, -bz, -bw end
	return bx, by, bz, bw
end

local function quatLog(x, y, z, w)
	local v2 = x * x + y * y + z * z
	if v2 < 1e-12 then return 0, 0, 0 end
	local v = math.sqrt(v2)
	local ang = 2.0 * math.atan2(v, w)
	local s = ang / v
	return x * s, y * s, z * s
end

-- ---------------------------------------------------------------------------
-- Sample pose at current animator state (root-space)
-- ---------------------------------------------------------------------------
local function sampleRootSpacePose(): { [string]: AnimTransform }
	local rootInv = hrp.CFrame:Inverse()
	local out: { [string]: AnimTransform } = {}
	for _, boneName in ipairs(R15_BONES) do
		local part = character:FindFirstChild(boneName)
		if part and part:IsA("BasePart") then
			local cf = rootInv * part.CFrame
			out[boneName] = { pos = cf.Position, rot = cf.Rotation }
		else
			out[boneName] = { pos = Vector3.zero, rot = CFrame.new() }
		end
	end
	return out
end

-- ---------------------------------------------------------------------------
-- Build ClipData: root-space pose at normalized times (no quantize for FFT)
-- ---------------------------------------------------------------------------
local function buildClipData(track: AnimationTrack, duration: number): (ClipData, number, { string })
	local numFrames = math.max(1, math.floor(NORMALIZE_DURATION * FPS + 0.5))
	local dt = NORMALIZE_DURATION / math.max(1, numFrames - 1)
	local out: ClipData = {}
	for _, bone in ipairs(R15_BONES) do
		out[bone] = table.create(numFrames)
	end

	for i = 0, numFrames - 1 do
		local t = i * dt
		local srcT = t * duration
		track.TimePosition = srcT
		animator:StepAnimations(0)
		local pose = sampleRootSpacePose()
		for _, bone in ipairs(R15_BONES) do
			out[bone][i + 1] = pose[bone]
		end
	end
	return out, numFrames, R15_BONES
end

-- ---------------------------------------------------------------------------
-- Motion signal: one real value per frame (like audio amplitude).
-- Sum over bones: position velocity magnitude + weight * rotation change (quat log).
-- ---------------------------------------------------------------------------
local MOTION_ROT_WEIGHT = 0.5

local function buildMotionSignal(cs: ClipData, tracks: { string }): { number }
	local n = #(cs[tracks[1]])
	local signal: { number } = table.create(n, 0)
	-- First frame: no velocity; use small constant so FFT doesn't see all zeros at start
	signal[1] = 1e-6

	for _, bone in ipairs(tracks) do
		local s = cs[bone]
		local qx0, qy0, qz0, qw0 = cframeToQuat(s[1].rot)
		local prevLx, prevLy, prevLz = 0, 0, 0
		for i = 2, n do
			local prev = s[i - 1]
			local curr = s[i]
			local dx = curr.pos.X - prev.pos.X
			local dy = curr.pos.Y - prev.pos.Y
			local dz = curr.pos.Z - prev.pos.Z
			local vel = math.sqrt(dx * dx + dy * dy + dz * dz)
			local qx, qy, qz, qw = cframeToQuat(curr.rot)
			qx, qy, qz, qw = quatHemisphereAlign(qx0, qy0, qz0, qw0, qx, qy, qz, qw)
			local lx, ly, lz = quatLog(qx, qy, qz, qw)
			local drot = math.sqrt((lx - prevLx)^2 + (ly - prevLy)^2 + (lz - prevLz)^2)
			prevLx, prevLy, prevLz = lx, ly, lz
			signal[i] = signal[i] + vel + MOTION_ROT_WEIGHT * drot
		end
	end
	return signal
end

-- ---------------------------------------------------------------------------
-- FFT: radix-2 Cooley-Tukey (in-place). Complex numbers as { re, im }.
-- ---------------------------------------------------------------------------
local function nextPowerOf2(n: number): number
	local p = 1
	while p < n do p *= 2 end
	return p
end

local function fftRadix2(x: { { number } }, inverse: boolean?)
	-- x[i] = { re, im }
	local N = #x
	if N <= 1 then return end
	local sign = (inverse == true) and 1 or -1
	local pi = math.pi

	-- Bit-reversal permutation
	local j = 0
	for i = 0, N - 1 do
		if i < j then
			x[i + 1], x[j + 1] = x[j + 1], x[i + 1]
		end
		local k = N // 2
		while k > 0 and k <= j do
			j -= k
			k //= 2
		end
		j += k
	end

	-- Cooley-Tukey
	local mm = 2
	while mm <= N do
		local wmRe = math.cos(sign * 2 * pi / mm)
		local wmIm = math.sin(sign * 2 * pi / mm)
		local half = mm // 2
		for k = 1, N, mm do
			local wRe, wIm = 1.0, 0.0
			for j = 0, half - 1 do
				local i1 = k + j
				local i2 = i1 + half
				local tRe = wRe * x[i2][1] - wIm * x[i2][2]
				local tIm = wRe * x[i2][2] + wIm * x[i2][1]
				x[i2][1] = x[i1][1] - tRe
				x[i2][2] = x[i1][2] - tIm
				x[i1][1] += tRe
				x[i1][2] += tIm
				local nwRe = wRe * wmRe - wIm * wmIm
				local nwIm = wRe * wmIm + wIm * wmRe
				wRe, wIm = nwRe, nwIm
			end
		end
		mm *= 2
	end

	if inverse then
		for i = 1, N do
			x[i][1] /= N
			x[i][2] /= N
		end
	end
end

-- Real signal -> FFT -> return magnitude spectrum (length N/2 + 1)
local function fftMagnitude(signal: { number }): { number }
	local N = #signal
	local complex: { { number } } = table.create(N)
	for i = 1, N do
		complex[i] = { signal[i], 0 }
	end
	fftRadix2(complex, false)
	local half = N // 2 + 1
	local mag: { number } = table.create(half)
	for i = 1, half do
		local re, im = complex[i][1], complex[i][2]
		mag[i] = math.sqrt(re * re + im * im)
	end
	return mag
end

-- ---------------------------------------------------------------------------
-- STFT: windowed FFT, return spectrogram [time_index][freq_bin] = log magnitude
-- ---------------------------------------------------------------------------
local function hannWindow(size: number): { number }
	local w: { number } = table.create(size)
	local pi = math.pi
	for i = 0, size - 1 do
		w[i + 1] = 0.5 * (1 - math.cos(2 * pi * i / (size - 1)))
	end
	return w
end

local function stft(signal: { number }, windowSize: number, hopSize: number): { { number } }
	local window = hannWindow(windowSize)
	local numFrames = math.floor((#signal - windowSize) / hopSize) + 1
	if numFrames <= 0 then return {} end

	local spectrogram: { { number } } = table.create(numFrames)
	for ti = 0, numFrames - 1 do
		local start = ti * hopSize + 1
		local framed: { number } = table.create(windowSize)
		for i = 1, windowSize do
			local idx = start + i - 1
			framed[i] = (signal[idx] or 0) * window[i]
		end
		local mag = fftMagnitude(framed)
		-- Log magnitude (like audio spectrogram)
		for i = 1, #mag do
			mag[i] = math.log(1 + mag[i])
		end
		spectrogram[ti + 1] = mag
	end
	return spectrogram
end

-- ---------------------------------------------------------------------------
-- Peak finding: local maxima in spectrogram (time x freq)
-- ---------------------------------------------------------------------------
local function findPeaks(spec: { { number } }, neighborhood: number): { { t: number, f: number, mag: number } }
	local peaks: { { t: number, f: number, mag: number } } = {}
	local nTime = #spec
	if nTime == 0 then return peaks end
	local nFreq = #spec[1]
	local r = neighborhood // 2  -- e.g. 3 -> r=1

	for ti = 1 + r, nTime - r do
		for fi = 1 + r, nFreq - r do
			local v = spec[ti][fi]
			local isMax = true
			for dt = -r, r do
				for df = -r, r do
					if (dt ~= 0 or df ~= 0) and spec[ti + dt][fi + df] >= v then
						isMax = false
						break
					end
				end
				if not isMax then break end
			end
			if isMax and v >= 0 then
				table.insert(peaks, { t = ti, f = fi, mag = v })
			end
		end
	end
	-- Fallback: if no local maxima (e.g. flat spectrum), take argmax per time frame so we still get hashes
	if #peaks < 2 then
		peaks = {}
		local nTime, nFreq = #spec, #spec[1]
		for ti = 1, nTime do
			local bestF, bestV = 1, 0
			for fi = 1, nFreq do
				local v = spec[ti][fi]
				if v > bestV then bestF = fi; bestV = v end
			end
			table.insert(peaks, { t = ti, f = bestF, mag = bestV })
		end
	end
	return peaks
end

-- ---------------------------------------------------------------------------
-- Constellation hashing (Shazam-style): each peak (t1,f1) pairs with up to
-- FAN_VALUE later peaks (t2,f2) with t2 - t1 in [1, MAX_DT]. Hash = f1, f2, dt.
-- Return list of { hash: number, t1: number }.
-- ---------------------------------------------------------------------------
local function constellationHashes(peaks: { { t: number, f: number, mag: number } }): { { hash: number, t1: number } }
	local hashes: { { hash: number, t1: number } } = {}
	-- Sort by time then by magnitude (desc) so we pick strong peaks first
	table.sort(peaks, function(a, b)
		if a.t ~= b.t then return a.t < b.t end
		return a.mag > b.mag
	end)

	for i = 1, #peaks do
		local p1 = peaks[i]
		local t1, f1 = p1.t, p1.f
		local count = 0
		for j = i + 1, #peaks do
			if count >= FAN_VALUE then break end
			local p2 = peaks[j]
			local dt = p2.t - t1
			if dt > MAX_DT then break end
			if dt >= 1 then
				local f2 = p2.f
				-- Encode (f1, f2, dt) into 32-bit hash (freqs typically < 128, dt < 16)
				local h1 = bit32.band(math.floor(f1), 0x7F)
				local h2 = bit32.lshift(bit32.band(math.floor(f2), 0x7F), 7)
				local h3 = bit32.lshift(bit32.band(math.floor(dt), 0xF), 14)
				local hash = bit32.bor(h1, bit32.bor(h2, h3))
				table.insert(hashes, { hash = hash, t1 = t1 })
				count = count + 1
			end
		end
	end
	return hashes
end

-- ---------------------------------------------------------------------------
-- Full pipeline: ClipData -> motion signal -> STFT -> peaks -> hashes
-- ---------------------------------------------------------------------------
local function buildFFTFingerprint(cs: ClipData, tracks: { string }): { { hash: number, t1: number } }
	local signal = buildMotionSignal(cs, tracks)
	-- Pad so STFT gives 4+ time frames (so findPeaks has 2+ time bins and constellation can form pairs with dt>=1)
	while #signal < WINDOW_SIZE + 3 * HOP_SIZE do
		table.insert(signal, 0)
	end
	local spec = stft(signal, WINDOW_SIZE, HOP_SIZE)
	local peaks = findPeaks(spec, PEAK_NEIGHBORHOOD)
	return constellationHashes(peaks)
end

-- ---------------------------------------------------------------------------
-- Main: walk clips -> build clip data -> FFT fingerprint -> write CSV
-- ---------------------------------------------------------------------------
local animation = Instance.new("Animation")
local header = "animId,clipId,duration,time_offset,hash\n"
local csv = header
local count = 0
local FLUSH_EVERY = 500

for fileData in FileSystemService:Walk(CLIPS_DIR, Enum.FileSystemWalkMode.NonRecursive) do
	local path = fileData.Path
	local instances = FileSystemService:LoadInstances(path)
	local clip = instances and instances[1]
	if not clip then continue end

	local ok, contentId = pcall(function()
		return KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
	end)
	if not ok or not contentId then
		clip:Destroy()
		continue
	end

	animation.AnimationId = contentId
	local track = animator:LoadAnimation(animation)
	track:Play(0)
	track.Looped = true
	wait(0)
	local duration = track.Length
	local animId, clipId = path:match("(%d+)%-(%d+)%.rbxm$")
	if not animId or not clipId then
		animId = "?"
		clipId = "?"
	end

	local success, err = pcall(function()
		local cs, numFrames, tracks = buildClipData(track, duration)
		local hashes = buildFFTFingerprint(cs, tracks)
		for _, h in ipairs(hashes) do
			csv = csv .. animId .. "," .. clipId .. "," .. string.format("%.6f", duration) .. "," .. tostring(h.t1) .. "," .. tostring(h.hash) .. "\n"
		end
	end)

	if not success then
		warn("FFT fingerprint failed:", path, err)
	end

	count = count + 1
	if count % 100 == 0 then print(count) end
	if count % FLUSH_EVERY == 0 and #csv > #header then
		FileSystemService:WriteFile(OUTPUT_CSV, csv, Enum.FileMode.Text)
	end

	track:Stop(0)
	track:Destroy()
	clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0)
		t:Destroy()
	end
	animator:StepAnimations(0)
end

FileSystemService:WriteFile(OUTPUT_CSV, csv, Enum.FileMode.Text)
print("Done. Wrote", count, "clips to", OUTPUT_CSV)
