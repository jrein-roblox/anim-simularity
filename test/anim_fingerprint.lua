--!strict
-- Pose-based animation fingerprinting: play clip via Animator, sample root-space
-- transforms, build 128D hand-crafted embedding for fuzzy similarity.
--
-- Run with roblox_cli (no asset-type branching; uses LoadAnimation + StepAnimations):
--   roblox-cli run --run <path>/anim_fingerprint.lua --fs.readwrite <path> --load.asRobloxScript
--
-- Requires: FStringAuthCookie if downloading assets. Output: fingerprints.csv in BASE_PATH.

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

-- Config: set to your repo path (used for clips dir and output CSV)
local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local OUTPUT_CSV = BASE_PATH .. "\\fingerprints.csv"
local FPS = 120
local NORMALIZE_DURATION = 1.0

-- ---------------------------------------------------------------------------
-- R15 character and animator
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

-- R15 bones (no HumanoidRootPart; we don't support root motion)
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
-- Math: quaternion, quantize, hash, embedding
-- ---------------------------------------------------------------------------
local function clamp(x, a, b) return math.max(a, math.min(b, x)) end

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

local function quatToCFrame(x: number, y: number, z: number, w: number): CFrame
	local len = math.sqrt(x * x + y * y + z * z + w * w)
	if len == 0 then return CFrame.new() end
	x, y, z, w = x / len, y / len, z / len, w / len
	local xx, yy, zz = x * x, y * y, z * z
	local xy, xz, yz = x * y, x * z, y * z
	local wx, wy, wz = w * x, w * y, w * z
	return CFrame.new(0, 0, 0,
		1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
		2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
		2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy))
end

local function quantizeFloat(x: number, q: number): number
	return math.floor(x / q + 0.5)
end

local U32 = 4294967296
local function u32(n: number): number return n % U32 end

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
-- Sample pose at current animator state: root-space transform for each bone
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
-- Build ClipData by stepping the track at normalized times (root-space, quantized)
-- ---------------------------------------------------------------------------
local quantPos = 1e-3
local quantQuat = 1e-4

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
			local fr = pose[bone]
			local qx, qy, qz, qw = cframeToQuat(fr.rot)
			qx = quantizeFloat(qx, quantQuat)
			qy = quantizeFloat(qy, quantQuat)
			qz = quantizeFloat(qz, quantQuat)
			qw = quantizeFloat(qw, quantQuat)
			local px = quantizeFloat(fr.pos.X, quantPos)
			local py = quantizeFloat(fr.pos.Y, quantPos)
			local pz = quantizeFloat(fr.pos.Z, quantPos)
			out[bone][i + 1] = {
				pos = Vector3.new(px, py, pz),
				rot = quatToCFrame(qx, qy, qz, qw),
			}
		end
	end
	return out, numFrames, R15_BONES
end

-- ---------------------------------------------------------------------------
-- 128D hand-crafted embedding (mean/var pos & quatLog, velocity, stillness per bone)
-- ---------------------------------------------------------------------------
local function buildEmbedding(cs: ClipData, tracks: { string }): { number }
	local feat: { number } = {}
	local function push(x: number) table.insert(feat, x) end
	local function push3(x: number, y: number, z: number)
		table.insert(feat, x); table.insert(feat, y); table.insert(feat, z)
	end

	for _, bone in ipairs(tracks) do
		local s = cs[bone]
		local n = s and #s or 0
		if n == 0 then
			for _ = 1, 16 do push(0.0) end
		else
			local qx0, qy0, qz0, qw0 = cframeToQuat(s[1].rot)
			local qlogs = table.create(n)
			local sumPX, sumPY, sumPZ = 0, 0, 0
			local sumLX, sumLY, sumLZ = 0, 0, 0
			for i = 1, n do
				local fr = s[i]
				sumPX += fr.pos.X; sumPY += fr.pos.Y; sumPZ += fr.pos.Z
				local qx, qy, qz, qw = cframeToQuat(fr.rot)
				qx, qy, qz, qw = quatHemisphereAlign(qx0, qy0, qz0, qw0, qx, qy, qz, qw)
				local lx, ly, lz = quatLog(qx, qy, qz, qw)
				qlogs[i] = { lx = lx, ly = ly, lz = lz }
				sumLX += lx; sumLY += ly; sumLZ += lz
			end
			local invN = 1.0 / n
			local meanPX, meanPY, meanPZ = sumPX * invN, sumPY * invN, sumPZ * invN
			local meanLX, meanLY, meanLZ = sumLX * invN, sumLY * invN, sumLZ * invN
			local varPX, varPY, varPZ = 0, 0, 0
			local varLX, varLY, varLZ = 0, 0, 0
			for i = 1, n do
				local fr = s[i]
				local dx = fr.pos.X - meanPX
				local dy = fr.pos.Y - meanPY
				local dz = fr.pos.Z - meanPZ
				varPX += dx * dx; varPY += dy * dy; varPZ += dz * dz
				local ql = qlogs[i]
				local dlx = ql.lx - meanLX
				local dly = ql.ly - meanLY
				local dlz = ql.lz - meanLZ
				varLX += dlx * dlx; varLY += dly * dly; varLZ += dlz * dlz
			end
			varPX *= invN; varPY *= invN; varPZ *= invN
			varLX *= invN; varLY *= invN; varLZ *= invN
			local velEX, velEY, velEZ = 0, 0, 0
			local accVar = 0.0
			if n > 1 then
				local prev = s[1].pos
				for i = 2, n do
					local p = s[i].pos
					local vx, vy, vz = p.X - prev.X, p.Y - prev.Y, p.Z - prev.Z
					prev = p
					velEX += math.abs(vx); velEY += math.abs(vy); velEZ += math.abs(vz)
					accVar += vx * vx + vy * vy + vz * vz
				end
				local invNSteps = 1.0 / (n - 1)
				velEX *= invNSteps; velEY *= invNSteps; velEZ *= invNSteps
			end
			-- Keep mean pose (6 dims) intact so different static poses stay distinguishable (avoids false positives)
			push3(meanPX, meanPY, meanPZ)
			push3(meanLX, meanLY, meanLZ)
			-- Single motion scalar per bone (variance + velocity + stillness) to stay within 128 dims
			local varPos = math.sqrt(varPX + varPY + varPZ)
			local varRot = math.sqrt(varLX + varLY + varLZ)
			local velMag = velEX + velEY + velEZ
			local still = 1.0 / (1.0 + accVar)
			table.insert(feat, varPos + varRot + velMag * 0.1 + still * 0.1)
		end
	end

	-- 7 per bone: 15*7 = 105. Pad to 128; fold overflow if ever > 128
	while #feat < 128 do table.insert(feat, 0.0) end
	if #feat > 128 then
		for i = 129, #feat do
			local j = ((i - 1) % 128) + 1
			feat[j] += feat[i]
		end
		for i = #feat, 129, -1 do table.remove(feat, i) end
	end
	local acc = 0.0
	for _, x in ipairs(feat) do acc += x * x end
	local inv = (acc > 0) and (1.0 / math.sqrt(acc)) or 1.0
	for i = 1, #feat do feat[i] *= inv end
	return feat
end

-- ---------------------------------------------------------------------------
-- Main: register clip -> LoadAnimation -> step -> root-space sample -> embed -> CSV
-- ---------------------------------------------------------------------------
local animation = Instance.new("Animation")
-- CSV: animId, clipId, duration, hash (0), emb1..emb128 (compatible with similarity.py)
local csv = "animId,clipId,duration,hash"
for i = 1, 128 do csv = csv .. ",emb" .. tostring(i) end
csv = csv .. "\n"
local count = 0
local FLUSH_EVERY = 2000

for fileData in FileSystemService:Walk(CLIPS_DIR, Enum.FileSystemWalkMode.NonRecursive) do
	local path = fileData.Path
	local instances = FileSystemService:LoadInstances(path)
	local clip = instances and instances[1]
	if not clip then continue end

	-- Single path: register clip and get a track (no asset-type switch)
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
	-- animId and clipId from path: e.g. .../12345-67890.rbxm
	local animId, clipId = path:match("(%d+)%-(%d+)%.rbxm$")
	if not animId or not clipId then
		animId = "?"
		clipId = "?"
	end

	local line = ""
	local success, err = pcall(function()
		local cs, numFrames, tracks = buildClipData(track, duration)
		local embedding = buildEmbedding(cs, tracks)
		line = animId .. "," .. clipId .. "," .. string.format("%.6f", duration) .. ",0"
		for _, v in ipairs(embedding) do
			line = line .. "," .. string.format("%.8f", v)
		end
		line = line .. "\n"
	end)

	if success and #line > 0 then
		csv = csv .. line
	else
		if not success then
			warn("Fingerprint failed:", path, err)
		end
	end

	count += 1
	if count % 100 == 0 then print(count) end
	if count % FLUSH_EVERY == 0 and #csv > 0 then
		FileSystemService:WriteFile(OUTPUT_CSV, csv, Enum.FileMode.Text)
	end

	-- Cleanup
	track:Stop(0)
	track:Destroy()
	clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0)
		t:Destroy()
	end
	animator:StepAnimations(0)
end

if #csv > 0 then
	FileSystemService:WriteFile(OUTPUT_CSV, csv, Enum.FileMode.Text)
end
print("Done. Wrote", count, "rows to", OUTPUT_CSV)
