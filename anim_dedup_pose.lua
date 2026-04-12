--!strict
--[[
  anim_dedup_pose.lua — Exact duplicate detection using Animator/AnimationTrack (pose-based)

  Same outputs as anim_dedup.lua, but poses a character by playing each clip via
  Animator:LoadAnimation, stepping TimePosition and StepAnimations, then sampling
  root-space bone transforms. Works with any clip type the engine can play
  (e.g. KeyframeSequence and any type accepted by the registration API), not only
  CurveAnimation.

  Walks out/clips/*.rbxm, registers each clip, plays it on an R15 character,
  samples poses at normalized times, hashes the canonical pose stream, and writes:
    hashes.csv   — animId, clipId, duration, hashFnv1a, hashMurmur
    dupes.csv    — groups of anim IDs (same FNV-1a hash)
    dupes2.csv   — groups (Murmur3 hash)
    dupes3.csv   — groups (combined 64-bit hash)

  Requirements:
    - Roblox CLI with --fs.readwrite. Clip files in out/clips/<animId>-<clipId>.rbxm.
    - Use download_assets.lua to download clips from a CSV first.

  Run:
    roblox-cli run --run <path>/anim_dedup_pose.lua --fs.readwrite <path> --load.asRobloxScript
]]

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

-- =============================================================================
-- CONFIG
-- =============================================================================
local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local HASHES_CSV = BASE_PATH .. "\\hashes_pose.csv"
local DUPES_FNV = BASE_PATH .. "\\dupes_pose.csv"
local DUPES_MURMUR = BASE_PATH .. "\\dupes_pose2.csv"
local DUPES_COMBINED = BASE_PATH .. "\\dupes_pose3.csv"

local FPS = 120
local NORMALIZE_DURATION = 1.0
local LOG_EVERY = 100

-- =============================================================================
-- R15 character and animator
-- =============================================================================
local function spawnR15(cframe: CFrame): Model
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "AnimDedupPoseR15"
	character.Parent = workspace
	if character.PrimaryPart then
		character:SetPrimaryPartCFrame(cframe)
	else
		local hrp = character:FindFirstChild("HumanoidRootPart")
		if hrp then (hrp :: BasePart).CFrame = cframe end
	end
	return character
end

local character = nil
local humanoid = nil
local animator = nil
local hrp = nil

-- R15 bones (no HumanoidRootPart; root-space pose, no root motion)
local R15_BONES = {
	"LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}

-- Instance parent for each bone (standard R15 rig); used for local-space sampling
local R15_BONE_PARENT: { [string]: string } = {
	LowerTorso = "HumanoidRootPart",
	UpperTorso = "LowerTorso",
	Head = "UpperTorso",
	LeftUpperArm = "UpperTorso",
	LeftLowerArm = "LeftUpperArm",
	LeftHand = "LeftLowerArm",
	RightUpperArm = "UpperTorso",
	RightLowerArm = "RightUpperArm",
	RightHand = "RightLowerArm",
	LeftUpperLeg = "LowerTorso",
	LeftLowerLeg = "LeftUpperLeg",
	LeftFoot = "LeftLowerLeg",
	RightUpperLeg = "LowerTorso",
	RightLowerLeg = "RightUpperLeg",
	RightFoot = "RightLowerLeg",
}

-- =============================================================================
-- Types
-- =============================================================================
type AnimTransform = { pos: Vector3, rot: CFrame }
type TrackFrames = { AnimTransform }
type ClipData = { [string]: TrackFrames }

-- =============================================================================
-- Math: quaternion, quantization, hashing
-- =============================================================================
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

local bit32 = bit32
local U32 = 4294967296

local function mul32(a: number, b: number): number
	local a_lo = bit32.band(a, 0xFFFF)
	local a_hi = bit32.rshift(a, 16)
	local b_lo = bit32.band(b, 0xFFFF)
	local b_hi = bit32.rshift(b, 16)
	local mid = (a_hi * b_lo + a_lo * b_hi) % 65536
	return ((a_lo * b_lo) + (mid * 65536)) % U32
end

local function writeUInt32LE(n: number): string
	local b = buffer.create(4)
	buffer.writeu32(b, 0, n % U32)
	return buffer.tostring(b)
end

local function fnv1a32(str: string): number
	local hash = 0x811C9DC5
	local FNV_PRIME = 16777619
	for i = 1, #str do
		hash = bit32.bxor(hash, string.byte(str, i))
		hash = mul32(hash, FNV_PRIME)
	end
	return hash
end

local function murmur3_32(str: string, seed: number?): number
	local c1 = 0xcc9e2d51
	local c2 = 0x1b873593
	local h1 = seed or 0
	local len = #str
	for i = 1, bit32.band(len, 0xFFFFFFFC), 4 do
		local k1 = string.unpack("<I4", str, i)
		k1 = mul32(k1, c1)
		k1 = bit32.lrotate(k1, 15)
		k1 = mul32(k1, c2)
		h1 = bit32.bxor(h1, k1)
		h1 = bit32.lrotate(h1, 13)
		h1 = (mul32(h1, 5) + 0xE6546B64) % U32
	end
	local tail_len = bit32.band(len, 3)
	local k1 = 0
	if tail_len >= 3 then k1 = bit32.bxor(k1, bit32.lshift(string.byte(str, len - 2), 16)) end
	if tail_len >= 2 then k1 = bit32.bxor(k1, bit32.lshift(string.byte(str, len - 1), 8)) end
	if tail_len >= 1 then
		k1 = bit32.bxor(k1, string.byte(str, len))
		k1 = mul32(k1, c1)
		k1 = bit32.lrotate(k1, 15)
		k1 = mul32(k1, c2)
		h1 = bit32.bxor(h1, k1)
	end
	h1 = bit32.bxor(h1, len)
	h1 = bit32.bxor(h1, bit32.rshift(h1, 16))
	h1 = mul32(h1, 0x85EBCA6B)
	h1 = bit32.bxor(h1, bit32.rshift(h1, 13))
	h1 = mul32(h1, 0xC2B2AE35)
	h1 = bit32.bxor(h1, bit32.rshift(h1, 16))
	return h1
end

-- =============================================================================
-- Sample root-space pose at current animator state
-- =============================================================================
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

-- Same bone order as sampleRootSpacePose, but each transform is relative to the bone's
-- parent BasePart (parent.CFrame:Inverse() * part.CFrame).
local function sampleLocalPose(): { [string]: AnimTransform }
	local out: { [string]: AnimTransform } = {}
	for _, boneName in ipairs(R15_BONES) do
		local parentName = R15_BONE_PARENT[boneName]
		local part = character:FindFirstChild(boneName)
		local parentPart = parentName and character:FindFirstChild(parentName)
		if part and part:IsA("BasePart") and parentPart and parentPart:IsA("BasePart") then
			local cf = parentPart.CFrame:Inverse() * part.CFrame
			out[boneName] = { pos = cf.Position, rot = cf.Rotation }
		else
			out[boneName] = { pos = Vector3.zero, rot = CFrame.new() }
		end
	end
	return out
end

-- =============================================================================
-- Build ClipData by stepping the track (root-space, quantized)
-- =============================================================================
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
		--local pose = sampleRootSpacePose()
		local pose = sampleLocalPose()
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
				pos = {px, py, pz},
				rot = {qx, qy, qz, qw},
				time = srcT,
			}
		end
	end
	return out, numFrames, R15_BONES
end

-- =============================================================================
-- Serialize ClipData to bytes (same order as anim_dedup for stable hash)
-- =============================================================================
local function serializeClip(cs: ClipData, tracks: { string }): string
	local chunks = {}
	for _, bone in ipairs(tracks) do
		local series = cs[bone]
		table.insert(chunks, writeUInt32LE(#series))
		for _, fr in ipairs(series) do
			table.insert(chunks, writeUInt32LE(fr.pos[1]))
			table.insert(chunks, writeUInt32LE(fr.pos[2]))
			table.insert(chunks, writeUInt32LE(fr.pos[3]))
			table.insert(chunks, writeUInt32LE(fr.rot[1]))
			table.insert(chunks, writeUInt32LE(fr.rot[2]))
			table.insert(chunks, writeUInt32LE(fr.rot[3]))
			table.insert(chunks, writeUInt32LE(fr.rot[4]))
		end
	end
	return table.concat(chunks)
end

local function hashClipData(cs: ClipData, tracks: { string }): (number, number)
	local bytes = serializeClip(cs, tracks)
	return fnv1a32(bytes), murmur3_32(bytes)
end

-- =============================================================================
-- Path and output helpers
-- =============================================================================
local function parseClipPath(path: string): (string?, string?)
	local basename = path:match("([^/\\]+)$") or path
	return basename:match("(%d+)%-(%d+)%.rbxm$")
end

local function writeDuplicateGroups(sortedGroups: { { string } }, outputPath: string)
	local lines = {}
	for _, group in ipairs(sortedGroups) do
		if #group > 1 then
			table.insert(lines, table.concat(group, ","))
		end
	end
	FileSystemService:WriteFile(outputPath, table.concat(lines, "\n"), Enum.FileMode.Text)
end

local function sortGroupsForOutput(hashmap: { [string]: { string } }): { { string } }
	local list = {}
	for _, group in pairs(hashmap) do
		table.sort(group, function(a, b) return (tonumber(a) or 0) < (tonumber(b) or 0) end)
		table.insert(list, group)
	end
	table.sort(list, function(a, b) return (tonumber(a[1]) or 0) < (tonumber(b[1]) or 0) end)
	return list
end

-- =============================================================================
-- Main: walk clips, play via Animator, hash pose data, write outputs
-- =============================================================================
print("anim_dedup_pose: scanning", CLIPS_DIR)
local hashmapFnv = {}
local hashmapMurmur = {}
local hashmapCombined = {}
local hashesCsvLines = {}
local count = 0
local skipped = 0

-- disable retargeting for now, we don't want to accound for changes it makes yet
game.Workspace.Retargeting = Enum.AnimatorRetargetingMode.Disabled
print(game.Workspace.Retargeting)

local animation = Instance.new("Animation")
local testData = {
		{Path = "C:/git/roblox/jrein/anim-simularity/out/clips/70369896323248-97913495804086.rbxm"},
		{Path = "C:/git/roblox/jrein/anim-simularity/out/clips/123790409181394-125606158001062.rbxm"},
}

--for _, fileData in ipairs(testData) do
for fileData in FileSystemService:Walk(CLIPS_DIR, Enum.FileSystemWalkMode.NonRecursive) do
	local path = fileData.Path
	local instances = FileSystemService:LoadInstances(path)
	local clip = instances and instances[1]

	-- for testing only
	-- if not clip or not clip:IsA("CurveAnimation") then
	-- 	skipped += 1
	-- 	if clip then clip:Destroy() end
	-- 	continue
	-- end

	if not clip then
		skipped += 1
		continue
	end

	local okRegister, contentId = pcall(function()
		return KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
	end)
	if not okRegister or not contentId then
		print(contentId)
		skipped += 1
		clip:Destroy()
		continue
	end

	-- TODO: we need to create a fresh character each clip cause there is
	-- something that doesn't reset the pose each time.
	character = spawnR15(CFrame.new(0, 0, 0))
	humanoid = character.Humanoid
	animator = humanoid.Animator
	hrp = character:FindFirstChild("HumanoidRootPart") :: BasePart

	animation.AnimationId = contentId
	local track = animator:LoadAnimation(animation)
	track:Play(0)
	track.Looped = true
	while track.Length == 0.0 do
		wait(0)
	end
	local duration = track.Length
	local animId, clipId = parseClipPath(path)
	if not animId or not clipId then
		track:Stop(0)
		track:Destroy()
		clip:Destroy()
		for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
			t:Stop(0)
			t:Destroy()
		end
		animator:StepAnimations(0)
		skipped += 1
		character:Destroy()
		continue
	end

	local okHash, err = pcall(function()
		local cs, _, tracks = buildClipData(track, duration)
		local clipHash, clipHash2 = hashClipData(cs, tracks)
		table.insert(hashesCsvLines, animId .. "," .. clipId .. "," .. string.format("%.6f", duration) .. "," .. tostring(clipHash) .. "," .. tostring(clipHash2))
		if hashmapFnv[clipHash] == nil then
			hashmapFnv[clipHash] = { animId }
		else
			table.insert(hashmapFnv[clipHash], animId)
		end
		if hashmapMurmur[clipHash2] == nil then
			hashmapMurmur[clipHash2] = { animId }
		else
			table.insert(hashmapMurmur[clipHash2], animId)
		end
		local key = string.format("%08X%08X", clipHash, clipHash2)
		if hashmapCombined[key] == nil then
			hashmapCombined[key] = { animId }
		else
			table.insert(hashmapCombined[key], animId)
		end

		--if animId == "100457501997256" or animId == "100887243791240" then
			-- print(path)
			-- local rig = clip:FindFirstChildOfClass("AnimationRigData")
			-- print(rig)

			-- print(clipHash, animId)
			-- for _, bone in ipairs(tracks) do
			-- 	local series = cs[bone]
			-- 	print(bone)
			-- 	if bone ~= "UpperTorso" then
			-- 		continue
			-- 	end
			-- 	for i, fr in ipairs(series) do
			-- 		print(i, fr.time, fr.pos[1], fr.pos[2], fr.pos[3], fr.rot[1], fr.rot[2], fr.rot[3], fr.rot[4])		
			-- 	end
			-- end
		--end

	end)

	track:Stop(0)
	track:Destroy()
	clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0)
		t:Destroy()
	end
	animator:StepAnimations(0)

	if not okHash then
		warn("Hash failed:", path, err)
		skipped += 1
		continue
	end

	count += 1
	if count % LOG_EVERY == 0 then
		print(count, "clips hashed")
	end

	-- nuke the character
	character:Destroy()
	humanoid = nil
	animator = nil
	hrp = nil

	-- if count > 1000 then
	-- 	break
	-- end
end

-- Write outputs
local hashesHeader = "animId,clipId,duration,hashFnv1a,hashMurmur\n"
FileSystemService:WriteFile(HASHES_CSV, hashesHeader .. table.concat(hashesCsvLines, "\n"), Enum.FileMode.Text)
print("Wrote", HASHES_CSV)

local sortedFnv = sortGroupsForOutput(hashmapFnv)
local sortedMurmur = sortGroupsForOutput(hashmapMurmur)
local sortedCombined = sortGroupsForOutput(hashmapCombined)

writeDuplicateGroups(sortedFnv, DUPES_FNV)
writeDuplicateGroups(sortedMurmur, DUPES_MURMUR)
writeDuplicateGroups(sortedCombined, DUPES_COMBINED)

local dupCountFnv = 0
local dupCountMurmur = 0
local dupCountCombined = 0
for _, g in ipairs(sortedFnv) do if #g > 1 then dupCountFnv += 1 end end
for _, g in ipairs(sortedMurmur) do if #g > 1 then dupCountMurmur += 1 end end
for _, g in ipairs(sortedCombined) do if #g > 1 then dupCountCombined += 1 end end

print("Wrote", DUPES_FNV, "| duplicate groups (FNV-1a):", dupCountFnv)
print("Wrote", DUPES_MURMUR, "| duplicate groups (Murmur3):", dupCountMurmur)
print("Wrote", DUPES_COMBINED, "| duplicate groups (combined):", dupCountCombined)
print("Done. Processed:", count, "| Skipped:", skipped)
