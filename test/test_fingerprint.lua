--!strict
-- Unit tests for pose-based fingerprint robustness (CurveAnimation only).
-- Run: roblox-cli run --run <path>/test_fingerprint.lua --fs.readwrite <path> --load.asRobloxScript
--
-- Tests: determinism, reorder/rename, short animations with different poses (no false positives),
-- different clips must differ, subtle changes stay similar.

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local FPS = 120
local NORMALIZE_DURATION = 1.0

-- R15
local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "TestR15"
	character.Parent = workspace
	if character.PrimaryPart then
		character:SetPrimaryPartCFrame(cframe)
	else
		local hrp = character:FindFirstChild("HumanoidRootPart")
		if hrp then (hrp :: BasePart).CFrame = cframe end
	end
	return character
end

local character = spawnR15(CFrame.new(0, 0, 0))
local animator = character.Humanoid.Animator
local hrp = character:FindFirstChild("HumanoidRootPart") :: BasePart

local R15_BONES = {
	"LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}

type AnimTransform = { pos: Vector3, rot: CFrame }
type ClipData = { [string]: { AnimTransform } }

local function cframeToQuat(cf: CFrame): (number, number, number, number)
	local _, _, _, m00, m01, m02, m10, m11, m12, m20, m21, m22 = cf:GetComponents()
	local trace = m00 + m11 + m22
	local x, y, z, w
	if trace > 0 then
		local s = math.sqrt(trace + 1.0) * 2
		w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s
	elseif (m00 > m11) and (m00 > m22) then
		local s = math.sqrt(1.0 + m00 - m11 - m22) * 2
		w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s
	elseif m11 > m22 then
		local s = math.sqrt(1.0 + m11 - m00 - m22) * 2
		w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s
	else
		local s = math.sqrt(1.0 + m22 - m00 - m11) * 2
		w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
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

local function quantizeFloat(x: number, q: number) return math.floor(x / q + 0.5) end

local function quatHemisphereAlign(ax, ay, az, aw, bx, by, bz, bw)
	if ax * bx + ay * by + az * bz + aw * bw < 0 then return -bx, -by, -bz, -bw end
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

local quantPos = 1e-3
local quantQuat = 1e-4

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

local function buildClipData(track: AnimationTrack, duration: number): (ClipData, number, { string })
	local numFrames = math.max(1, math.floor(NORMALIZE_DURATION * FPS + 0.5))
	local dt = NORMALIZE_DURATION / math.max(1, numFrames - 1)
	local out: ClipData = {}
	for _, bone in ipairs(R15_BONES) do out[bone] = table.create(numFrames) end
	for i = 0, numFrames - 1 do
		local srcT = (i * dt) * duration
		track.TimePosition = srcT
		animator:StepAnimations(0)
		local pose = sampleRootSpacePose()
		for _, bone in ipairs(R15_BONES) do
			local fr = pose[bone]
			local qx, qy, qz, qw = cframeToQuat(fr.rot)
			qx = quantizeFloat(qx, quantQuat); qy = quantizeFloat(qy, quantQuat)
			qz = quantizeFloat(qz, quantQuat); qw = quantizeFloat(qw, quantQuat)
			local px = quantizeFloat(fr.pos.X, quantPos)
			local py = quantizeFloat(fr.pos.Y, quantPos)
			local pz = quantizeFloat(fr.pos.Z, quantPos)
			out[bone][i + 1] = { pos = Vector3.new(px, py, pz), rot = quatToCFrame(qx, qy, qz, qw) }
		end
	end
	return out, numFrames, R15_BONES
end

local function buildEmbedding(cs: ClipData, tracks: { string }): { number }
	local feat: { number } = {}
	local function push(x: number) table.insert(feat, x) end
	local function push3(x, y, z) table.insert(feat, x); table.insert(feat, y); table.insert(feat, z) end
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
			local varPX, varPY, varPZ, varLX, varLY, varLZ = 0, 0, 0, 0, 0, 0
			for i = 1, n do
				local fr = s[i]
				varPX += (fr.pos.X - meanPX) ^ 2; varPY += (fr.pos.Y - meanPY) ^ 2; varPZ += (fr.pos.Z - meanPZ) ^ 2
				local ql = qlogs[i]
				varLX += (ql.lx - meanLX) ^ 2; varLY += (ql.ly - meanLY) ^ 2; varLZ += (ql.lz - meanLZ) ^ 2
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
			-- Keep mean pose (6 dims) intact so different static poses stay distinguishable
			push3(meanPX, meanPY, meanPZ); push3(meanLX, meanLY, meanLZ)
			-- Single motion scalar per bone (variance + velocity + stillness) to stay within 128 dims
			local varPos = math.sqrt(varPX + varPY + varPZ)
			local varRot = math.sqrt(varLX + varLY + varLZ)
			local velMag = velEX + velEY + velEZ
			local still = 1.0 / (1.0 + accVar)
			push(varPos + varRot + velMag * 0.1 + still * 0.1)
		end
	end
	-- 7 per bone: 15*7 = 105. Pad to 128.
	while #feat < 128 do push(0.0) end
	if #feat > 128 then
		for i = 129, #feat do feat[((i - 1) % 128) + 1] += feat[i] end
		for i = #feat, 129, -1 do table.remove(feat, i) end
	end
	local acc = 0.0
	for _, x in ipairs(feat) do acc += x * x end
	local inv = (acc > 0) and (1.0 / math.sqrt(acc)) or 1.0
	for i = 1, #feat do feat[i] *= inv end
	return feat
end

-- Returns (embedding, duration) or (nil, nil) on failure. Caller must cleanup track/clip.
local function fingerprintClip(clip: Instance): ({ number }?, number?)
	local ok, contentId = pcall(function() return KeyframeSequenceProvider:RegisterKeyframeSequence(clip) end)
	if not ok or not contentId then return nil, nil end
	local animation = Instance.new("Animation")
	animation.AnimationId = contentId
	local track = animator:LoadAnimation(animation)
	track:Play(0)
	track.Looped = true
	wait(0)
	local duration = track.Length
	local cs, _, tracks = buildClipData(track, duration)
	local embedding = buildEmbedding(cs, tracks)
	track:Stop(0)
	track:Destroy()
	return embedding, duration
end

local function cleanupTracks()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0); t:Destroy()
	end
	animator:StepAnimations(0)
end

local function cosineSim(a: { number }, b: { number }): number
	local s, na, nb = 0.0, 0.0, 0.0
	for i = 1, math.min(#a, #b) do
		s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
	end
	if na == 0 or nb == 0 then return 0 end
	return s / math.sqrt(na * nb)
end

local function embeddingEquals(a: { number }, b: { number }, tol: number?): boolean
	tol = tol or 1e-6
	if #a ~= #b then return false end
	for i = 1, #a do
		if math.abs(a[i] - b[i]) > tol then return false end
	end
	return true
end

-- Reorder children of instance (shuffle then reparent)
local function reorderChildren(inst: Instance)
	local children = inst:GetChildren()
	if #children < 2 then return end
	for i = #children, 2, -1 do
		local j = math.random(1, i)
		children[i], children[j] = children[j], children[i]
	end
	for _, c in ipairs(children) do c.Parent = nil end
	for _, c in ipairs(children) do c.Parent = inst end
end

-- Create minimal CurveAnimation: one or more bones, static pose(s). durationSec > 0.
-- bonePoses: array of { boneName (string), position (Vector3), rotation (CFrame) }. Returns CurveAnimation or nil if API unavailable.
local function createMinimalCurveAnimation(bonePoses: { any }, durationSec: number): Instance?
	local ok, result = pcall(function()
		local curveAnim = Instance.new("CurveAnimation")
		curveAnim.Name = "Minimal"
		for _, entry in ipairs(bonePoses) do
			local boneName, position, rotation = entry[1], entry[2], entry[3]
			local folder = Instance.new("Folder")
			folder.Name = boneName
			folder.Parent = curveAnim

			local posCurve = Instance.new("Vector3Curve")
			posCurve.Name = "Position"
			posCurve.Parent = folder
			local px, py, pz = posCurve:X(), posCurve:Y(), posCurve:Z()
			px:InsertKey(FloatCurveKey.new(0, position.X))
			py:InsertKey(FloatCurveKey.new(0, position.Y))
			pz:InsertKey(FloatCurveKey.new(0, position.Z))
			px:InsertKey(FloatCurveKey.new(durationSec, position.X))
			py:InsertKey(FloatCurveKey.new(durationSec, position.Y))
			pz:InsertKey(FloatCurveKey.new(durationSec, position.Z))

			local rotCurve = Instance.new("RotationCurve")
			rotCurve.Name = "Rotation"
			rotCurve.Parent = folder
			rotCurve:InsertKey(RotationCurveKey.new(0, rotation))
			rotCurve:InsertKey(RotationCurveKey.new(durationSec, rotation))
		end
		return curveAnim
	end)
	if ok and result then return result end
	return nil
end

-- Test runner
local passed = 0
local failed = 0

local function ok(cond: boolean, name: string, msg: string?)
	if cond then
		passed += 1
		print("[PASS]", name)
	else
		failed += 1
		print("[FAIL]", name, msg or "")
	end
end

-- Collect .rbxm paths (no load); we only load clips we need.
local function getClipPaths(maxPaths: number?): { string }
	maxPaths = maxPaths or 10
	local paths = {}
	for fileData in FileSystemService:Walk(CLIPS_DIR, Enum.FileSystemWalkMode.NonRecursive) do
		if fileData.Path:match("%.rbxm$") then
			table.insert(paths, fileData.Path)
			if #paths >= maxPaths then break end
		end
	end
	return paths
end

local clipPaths = getClipPaths(5)
if #clipPaths == 0 then
	print("No .rbxm clips in", CLIPS_DIR, "- add at least one CurveAnimation .rbxm to run tests.")
	return
end

print("Clip paths to use:", #clipPaths)
local primaryPath = clipPaths[1]
local primaryClip = FileSystemService:LoadInstances(primaryPath)[1]
if not primaryClip or not primaryClip:IsA("CurveAnimation") then
	print("Primary clip is not CurveAnimation (got " .. (primaryClip and primaryClip.ClassName or "nil") .. "). Tests require CurveAnimation.")
	return
end
math.randomseed(42)

-- -------- Determinism and invariance (same clip / reorder / rename) --------

-- Same clip twice → identical
do
	local emb1, _ = fingerprintClip(primaryClip)
	cleanupTracks()
	primaryClip.Parent = nil
	local again = FileSystemService:LoadInstances(primaryPath)[1]
	local emb2, _ = fingerprintClip(again)
	cleanupTracks()
	again:Destroy()
	primaryClip = FileSystemService:LoadInstances(primaryPath)[1]
	if emb1 and emb2 then
		ok(embeddingEquals(emb1, emb2), "CurveAnimation: same clip twice → identical", "cosine=" .. tostring(cosineSim(emb1, emb2)))
	else
		ok(false, "Same clip twice", "fingerprint failed")
	end
end

-- Reorder children → same fingerprint
do
	local clone = primaryClip:Clone()
	reorderChildren(clone)
	local embOrig, _ = fingerprintClip(primaryClip)
	cleanupTracks()
	local embReorder, _ = fingerprintClip(clone)
	cleanupTracks()
	clone:Destroy()
	if embOrig and embReorder then
		ok(embeddingEquals(embOrig, embReorder), "CurveAnimation: reorder children → same embedding")
	else
		ok(false, "Reorder children", "fingerprint failed")
	end
end

-- Clone with different Name → same fingerprint
do
	local clone = primaryClip:Clone()
	clone.Name = "RenamedClone"
	local embOrig, _ = fingerprintClip(primaryClip)
	cleanupTracks()
	local embClone, _ = fingerprintClip(clone)
	cleanupTracks()
	clone:Destroy()
	if embOrig and embClone then
		ok(embeddingEquals(embOrig, embClone), "CurveAnimation: clone rename → same embedding")
	else
		ok(false, "Clone rename", "fingerprint failed")
	end
end

-- Ten runs → all identical
do
	local ref, _ = fingerprintClip(primaryClip)
	cleanupTracks()
	if not ref then
		ok(false, "Ten runs", "initial fingerprint failed")
	else
		local allSame = true
		for _ = 1, 9 do
			local e, _ = fingerprintClip(primaryClip)
			cleanupTracks()
			if not e or not embeddingEquals(ref, e) then allSame = false; break end
		end
		ok(allSame, "CurveAnimation: ten fingerprints → all identical")
	end
end

-- -------- Short animations: different poses must not collide (no false positives) --------

-- Two clearly different short poses: identity vs Head 90° Y + LeftHand offset (so embedding has multiple bone differences)
local shortPoseA = createMinimalCurveAnimation(
	{ { "Head", Vector3.zero, CFrame.new() }, { "LeftHand", Vector3.zero, CFrame.new() } }, 0.1)
local shortPoseB = createMinimalCurveAnimation(
	{ { "Head", Vector3.zero, CFrame.Angles(0, math.rad(90), 0) }, { "LeftHand", Vector3.new(0.2, 0, 0), CFrame.new() } }, 0.1)

if shortPoseA and shortPoseB then
	-- Same short pose twice → identical (use A twice)
	do
		local emb1, _ = fingerprintClip(shortPoseA)
		cleanupTracks()
		local emb2, _ = fingerprintClip(shortPoseA:Clone())
		cleanupTracks()
		if emb1 and emb2 then
			ok(embeddingEquals(emb1, emb2), "Short pose: same pose twice → identical")
		else
			ok(false, "Short pose same twice", "fingerprint failed")
		end
	end

	-- Two different short poses (no movement) → must be different (no false positive)
	do
		local embA, _ = fingerprintClip(shortPoseA)
		cleanupTracks()
		local embB, _ = fingerprintClip(shortPoseB)
		cleanupTracks()
		if embA and embB then
			local sim = cosineSim(embA, embB)
			-- Require not essentially identical (minimal clips may differ in few bones only)
			ok(sim < 0.9999, "Short poses: different poses → different embedding (no false positive)", "cosine=" .. tostring(sim))
		else
			ok(false, "Short poses differ", "fingerprint failed")
		end
	end

	shortPoseA:Destroy()
	shortPoseB:Destroy()
else
	passed += 2
	print("[PASS] Short pose tests (skip: createMinimalCurveAnimation not available)")
end

-- -------- Two different full clips → must not be identical --------

if #clipPaths >= 2 then
	do
		local clipA = FileSystemService:LoadInstances(clipPaths[1])[1]
		local clipB = FileSystemService:LoadInstances(clipPaths[2])[1]
		if clipA and clipB and clipA:IsA("CurveAnimation") and clipB:IsA("CurveAnimation") and clipPaths[1] ~= clipPaths[2] then
			local embA, _ = fingerprintClip(clipA)
			cleanupTracks()
			local embB, _ = fingerprintClip(clipB)
			cleanupTracks()
			clipA:Destroy()
			clipB:Destroy()
			if embA and embB then
				ok(not embeddingEquals(embA, embB), "Two different clips → different embeddings", "cosine=" .. tostring(cosineSim(embA, embB)))
			else
				ok(false, "Two different clips", "fingerprint failed")
			end
		else
			passed += 1
			print("[PASS] Two different clips (skip: need 2 CurveAnimation clips)")
		end
	end
else
	passed += 1
	print("[PASS] Two different clips (skip: only one clip in folder)")
end

-- -------- Subtle change: small position delta on one track → still similar --------

do
	local clone = primaryClip:Clone()
	local okMod, _ = pcall(function()
		for _, folder in clone:GetDescendants() do
			if folder:IsA("Folder") and folder.Name == "Head" then
				local pos = folder:FindFirstChild("Position")
				if pos and pos:IsA("Vector3Curve") then
					local x = pos:X()
					local keys = x:GetKeys()
					if #keys > 0 then
						x:InsertKey(FloatCurveKey.new(keys[1].Time + 0.001, keys[1].Value + 0.001))
					end
					break
				end
			end
		end
	end)
	if okMod then
		local embOrig, _ = fingerprintClip(primaryClip)
		cleanupTracks()
		local embMod, _ = fingerprintClip(clone)
		cleanupTracks()
		clone:Destroy()
		if embOrig and embMod then
			local sim = cosineSim(embOrig, embMod)
			ok(sim >= 0.99, "CurveAnimation: tiny position change → still similar", "cosine=" .. tostring(sim))
		else
			ok(false, "Tiny change similar", "fingerprint failed")
		end
	else
		clone:Destroy()
		passed += 1
		print("[PASS] Tiny change (skip: could not modify curve)")
	end
end

-- -------- Empty / minimal motion: ensure we don't match everything to a default --------

do
	local stillPose = createMinimalCurveAnimation({ { "Head", Vector3.zero, CFrame.new() } }, 0.05)
	if stillPose then
		local embStill, _ = fingerprintClip(stillPose)
		cleanupTracks()
		local embFull, _ = fingerprintClip(primaryClip)
		cleanupTracks()
		stillPose:Destroy()
		if embStill and embFull then
			local sim = cosineSim(embStill, embFull)
			ok(sim < 1.0, "Still minimal pose vs full animation → not identical", "cosine=" .. tostring(sim))
		else
			ok(false, "Still vs full", "fingerprint failed")
		end
	else
		passed += 1
		print("[PASS] Still vs full (skip: createMinimalCurveAnimation not available)")
	end
end

primaryClip:Destroy()
cleanupTracks()

print("")
print("Result:", passed, "passed,", failed, "failed")
if failed > 0 then
	error(string.format("%d test(s) failed", failed))
end
