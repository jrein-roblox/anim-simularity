--!strict
--[[
  anim_dedup.lua — Exact duplicate detection for Roblox CurveAnimation clips

  Walks a folder of saved clip files (out/clips/*.rbxm), hashes each CurveAnimation
  by canonicalizing pose data (normalized duration, fixed FPS, quantized pos/rot),
  then groups clips that share the same hash and writes duplicate lists to CSV.

  Outputs:
    hashes.csv   — animId, clipId, duration, hashFnv1a, hashMurmur (one row per clip)
    dupes.csv    — groups of anim IDs that share the same FNV-1a hash (exact duplicates)
    dupes2.csv   — same but for Murmur3 hash
    dupes3.csv   — same but for combined 64-bit hash (fewest false collisions)

  Requirements:
    - Roblox CLI (roblox-cli) with --fs.readwrite so the script can read/write files.
    - Clip files already on disk: out/clips/<animId>-<clipId>.rbxm (CurveAnimation only).
    - To download clips first, use download_assets.lua with a CSV of catalog IDs.

  Run:
    roblox-cli run --run <path>/anim_dedup.lua --fs.readwrite <path> --load.asRobloxScript

  Example:
    roblox-cli run --run C:/repo/anim-simularity/anim_dedup.lua --fs.readwrite C:/repo/anim-simularity --load.asRobloxScript
]]

local FileSystemService = game:GetService("FileSystemService")
local InsertService = game:GetService("InsertService")

-- =============================================================================
-- CONFIG — set paths and options here
-- =============================================================================
local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local HASHES_CSV = BASE_PATH .. "\\hashes.csv"
local DUPES_FNV = BASE_PATH .. "\\dupes.csv"
local DUPES_MURMUR = BASE_PATH .. "\\dupes2.csv"
local DUPES_COMBINED = BASE_PATH .. "\\dupes3.csv"

local FPS = 60
local DURATION_MODE = "NormalizeTo1"
local LOG_EVERY = 100


-- =============================================================================
-- Types
-- =============================================================================
export type AnimTransform = { pos: Vector3, rot: CFrame }
export type TrackFrames = { AnimTransform }
export type ClipData = { [string]: TrackFrames }
export type FingerprintParams = {
	FPS: number?,
	DurationMode: "NormalizeTo1" | "Preserve"?,
	QuantPos: number?,
	QuantQuat: number?,
	TrackOrder: { string }?,
}

-- =============================================================================
-- CurveAnimation: read tracks and sample at time
-- =============================================================================
local function getCurveTracks(curve: CurveAnimation): { [string]: { pos: Instance, rot: Instance } }
	local tracks = {}
	for _, des in ipairs(curve:GetDescendants()) do
		if des:IsA("Folder") and des:FindFirstChild("Position") and des:FindFirstChild("Rotation") then
			tracks[des.Name] = {
				pos = des:FindFirstChild("Position") :: Instance,
				rot = des:FindFirstChild("Rotation") :: Instance,
			}
		end
	end
	return tracks
end

local function getCurveValueAtTime(curve: Instance, time: number): any
	if curve:IsA("FloatCurve") then
		return curve:GetValueAtTime(time)
	elseif curve:IsA("Vector3Curve") then
		local v = curve:GetValueAtTime(time)
		return Vector3.new(v[1], v[2], v[3])
	elseif curve:IsA("EulerRotationCurve") then
		return curve:GetRotationAtTime(time)
	elseif curve:IsA("RotationCurve") then
		return curve:GetValueAtTime(time)
	elseif curve:IsA("MarkerCurve") then
		local markers = curve:GetMarkers()
		for i = #markers, 1, -1 do
			if markers[i].Time <= time then
				return markers[i].Value
			end
		end
		return nil
	else
		warn("Unrecognized curve type:", curve.ClassName)
		return nil
	end
end

local function calculateCurveAnimLength(curveAnim: Instance): number
	local tracks = getCurveTracks(curveAnim :: CurveAnimation)
	local maxTime = -1
	local function scanFloatKeys(container: Instance?)
		if not container then return end
		for _, child in container:GetChildren() do
			if not child:IsA("FloatCurve") then continue end
			for _, key in child:GetKeys() do
				maxTime = math.max(maxTime, key.Time)
			end
		end
	end
	for _, t in pairs(tracks) do
		scanFloatKeys(t.pos)
		scanFloatKeys(t.rot)
	end
	return maxTime
end

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

-- =============================================================================
-- Canonicalize clip and hash
-- =============================================================================
local R15_BONE_NAMES = {
	"HumanoidRootPart", "LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}

local function buildTrackList(explicitOrder: { string }?): { string }
	if explicitOrder and #explicitOrder > 0 then
		return table.clone(explicitOrder)
	end
	return table.clone(R15_BONE_NAMES)
end

local function sampleTrackTransform(track: any, t: number): AnimTransform
	if not track then
		return { pos = Vector3.zero, rot = CFrame.new() }
	end
	return {
		pos = getCurveValueAtTime(track.pos, t),
		rot = getCurveValueAtTime(track.rot, t),
	}
end

local function canonicalize(curveAnim: Instance, duration: number, params: FingerprintParams?): (ClipData, number, { string })
	params = params or {}
	local fps = params.FPS or 60
	local quantPos = params.QuantPos or 0.01
	local quantQuat = params.QuantQuat or 0.01
	local durationMode = params.DurationMode or "NormalizeTo1"

	local tracks = buildTrackList(params.TrackOrder)
	local tracksCurves = getCurveTracks(curveAnim :: CurveAnimation)
	local out: ClipData = {}

	local normDuration = (durationMode == "NormalizeTo1") and 1.0 or duration
	local numFrames = math.max(1, math.floor(normDuration * fps + 0.5))
	local dt = normDuration / math.max(1, numFrames - 1)

	for _, bone in ipairs(tracks) do
		local frames: TrackFrames = table.create(numFrames)
		for i = 0, numFrames - 1 do
			local t = i * dt
			local srcT = (durationMode == "NormalizeTo1") and (t * duration) or t
			local trackCurve = tracksCurves[bone]
			local fr = sampleTrackTransform(trackCurve, srcT)
			local qx, qy, qz, qw = cframeToQuat(fr.rot)
			qx = quantizeFloat(qx, quantQuat)
			qy = quantizeFloat(qy, quantQuat)
			qz = quantizeFloat(qz, quantQuat)
			qw = quantizeFloat(qw, quantQuat)
			local px = quantizeFloat(fr.pos.X, quantPos)
			local py = quantizeFloat(fr.pos.Y, quantPos)
			local pz = quantizeFloat(fr.pos.Z, quantPos)
			frames[i + 1] = {
				pos = {px, py, pz},
				rot = {qx, qy, qz, qw},
				time = srcT,
			}
		end
		out[bone] = frames
	end
	return out, numFrames, tracks
end

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

local function hashAnimation(curveAnim: Instance, duration: number, params: FingerprintParams?): (number, number)
	local cs, _, tracks = canonicalize(curveAnim, duration, params)
	local bytes = serializeClip(cs, tracks)
	return fnv1a32(bytes), murmur3_32(bytes), cs,tracks
end

-- =============================================================================
-- Extract animId and clipId from clip file path (works with / or \)
-- =============================================================================
local function parseClipPath(path: string): (string?, string?)
	local basename = path:match("([^/\\]+)$") or path
	return basename:match("(%d+)%-(%d+)%.rbxm$")
end

-- =============================================================================
-- Write duplicate groups to a file (sorted by group, then by anim ID)
-- =============================================================================
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
-- Main: walk clips, hash, aggregate, write outputs
-- =============================================================================
print("anim_dedup: scanning", CLIPS_DIR)
local hashmapFnv = {}
local hashmapMurmur = {}
local hashmapCombined = {}
local hashesCsvLines = {}
local count = 0
local skipped = 0

for fileData in FileSystemService:Walk(CLIPS_DIR, Enum.FileSystemWalkMode.NonRecursive) do
	local path = fileData.Path
	local instances = FileSystemService:LoadInstances(path)
	local clip = instances and instances[1]

	if not clip or not clip:IsA("CurveAnimation") then
		skipped += 1
		if clip then clip:Destroy() end
		continue
	end

	local animId, clipId = parseClipPath(path)
	if not animId or not clipId then
		clip:Destroy()
		continue
	end

	local ok, err = pcall(function()
		local duration = calculateCurveAnimLength(clip)
		local clipHash, clipHash2, cs, tracks = hashAnimation(clip, duration, {
			FPS = FPS,
			DurationMode = DURATION_MODE,
		})

		-- hashes.csv row
		table.insert(hashesCsvLines, animId .. "," .. clipId .. "," .. string.format("%.6f", duration) .. "," .. tostring(clipHash) .. "," .. tostring(clipHash2))

		-- Group by FNV-1a
		if hashmapFnv[clipHash] == nil then
			hashmapFnv[clipHash] = { animId }
		else
			table.insert(hashmapFnv[clipHash], animId)
		end

		-- Group by Murmur3
		if hashmapMurmur[clipHash2] == nil then
			hashmapMurmur[clipHash2] = { animId }
		else
			table.insert(hashmapMurmur[clipHash2], animId)
		end

		-- Group by combined 64-bit
		local key = string.format("%08X%08X", clipHash, clipHash2)
		if hashmapCombined[key] == nil then
			hashmapCombined[key] = { animId }
		else
			table.insert(hashmapCombined[key], animId)
		end

		-- if animId == "100457501997256" or animId == "100887243791240" then
		-- 	print(clipHash, animId)
		-- 	for _, bone in ipairs(tracks) do
		-- 		local series = cs[bone]
		-- 		if bone ~= "LeftHand" then
		-- 			continue
		-- 		end
		-- 		print(bone)
		-- 		for i, fr in ipairs(series) do
		-- 			print(i, fr.time, fr.pos[1], fr.pos[2], fr.pos[3], fr.rot[1], fr.rot[2], fr.rot[3], fr.rot[4])		
		-- 		end
		-- 	end
		-- end
	end)

	if not ok then
		warn("Hash failed:", path, err)
	end

	count += 1
	if count % LOG_EVERY == 0 then
		print(count, "clips hashed")
	end
	clip:Destroy()
end

-- Write hashes.csv (header + rows)
local hashesHeader = "animId,clipId,duration,hashFnv1a,hashMurmur\n"
FileSystemService:WriteFile(HASHES_CSV, hashesHeader .. table.concat(hashesCsvLines, "\n"), Enum.FileMode.Text)
print("Wrote", HASHES_CSV)

-- Sort and write duplicate groups
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
print("Done. Processed:", count, "| Skipped (non-CurveAnimation):", skipped)
