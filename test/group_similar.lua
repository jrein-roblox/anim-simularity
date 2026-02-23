--!strict
-- Build groups of similar animations: (1) fingerprint cosine similarity, (2) detailed
-- pose comparison on each similar pair; only pairs that pass both become edges.
-- Output: smaller groups of actual duplicates; non-duplicates are discarded from multi-anim groups.
--
-- Reads animId, clipId, duration, hash, emb1..emb128; cosine threshold -> candidate edges;
-- for each candidate edge, loads both clips and compares pose stream; only if duplicate
-- score >= DETAIL_DUP_THRESHOLD is the edge kept. Writes groups.csv and groups.txt.
--
-- Run: roblox-cli run --run <path>/group_similar.lua --fs.readwrite <path> --load.asRobloxScript

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

-- Config (relative paths work when run via roblox-cli --fs.readwrite <repo>)
local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local INPUT_CSV = BASE_PATH .. "\\fingerprints.csv"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local OUTPUT_PREFIX = BASE_PATH .. "\\groups"
local INPUT_CSV_REL = "fingerprints.csv"
local CLIPS_DIR_REL = "out\\clips"
local OUTPUT_PREFIX_REL = "groups"
local THRESHOLD = 0.99
local DETAIL_DUP_THRESHOLD = 0.7  -- duplicate score (0-1) above this = keep edge
local SKIP_DETAILED_VERIFY = false -- if true, use cosine edges only (no clip load)
local FPS = 120
local NORMALIZE_DURATION = 1.0
local VERIFY_FPS = 30   -- fewer frames for detailed check = faster
local VERIFY_DURATION = 1.0

local EMB_DIM = 128
local EMB_START_COL = 5   -- column 5 = emb1 (after animId, clipId, duration, hash)

-- ---------------------------------------------------------------------------
-- CSV parse
-- ---------------------------------------------------------------------------
local function parseCSV(csvData: string, sep: string?, quote: string?): { { string } }
	sep = sep or ","
	quote = quote or '"'
	csvData = csvData:gsub("\r\n", "\n"):gsub("\r", "\n")
	local rows: { { string } } = {}
	local row: { string } = {}
	local fieldBuf: { string } = {}
	local n = #csvData
	local i = 1
	local inQuotes = false

	local function pushField()
		table.insert(row, table.concat(fieldBuf))
		table.clear(fieldBuf)
	end
	local function pushRow()
		pushField()
		table.insert(rows, row)
		row = {}
	end

	while i <= n do
		local c = string.sub(csvData, i, i)
		if inQuotes then
			if c == quote then
				local nxt = string.sub(csvData, i + 1, i + 1)
				if nxt == quote then
					table.insert(fieldBuf, quote)
					i += 2
				else
					inQuotes = false
					i += 1
				end
			else
				table.insert(fieldBuf, c)
				i += 1
			end
		else
			if c == quote then
				inQuotes = true
				i += 1
			elseif c == sep then
				pushField()
				i += 1
			elseif c == "\n" then
				pushRow()
				i += 1
			else
				table.insert(fieldBuf, c)
				i += 1
			end
		end
	end
	if #fieldBuf > 0 or #row > 0 then
		pushRow()
	end
	return rows
end

-- ---------------------------------------------------------------------------
-- Load fingerprints: animIds, clipIds (arrays same order), durations (map), embeddings
-- ---------------------------------------------------------------------------
local function loadFingerprints(path: string): ({ string }, { string }, { [string]: number }, { { number } })
	local animIds: { string } = {}
	local clipIds: { string } = {}
	local durations: { [string]: number } = {}
	local embList: { { number } } = {}

	local ok, csvData = pcall(function()
		return FileSystemService:ReadFile(path, Enum.FileMode.Text)
	end)
	if not ok or not csvData then
		return animIds, clipIds, durations, embList
	end

	local rows = parseCSV(csvData)
	local startRow = 1
	if #rows >= 1 and not tonumber(rows[1][1]) then
		startRow = 2
	end

	for r = startRow, #rows do
		local row = rows[r]
		if not row or #row < EMB_START_COL + EMB_DIM - 1 then
			continue
		end
		local aid = row[1]
		table.insert(animIds, aid)
		table.insert(clipIds, row[2] or "")
		local dur = tonumber(row[3])
		durations[aid] = (dur and dur) or 0
		local emb: { number } = {}
		for k = 1, EMB_DIM do
			local v = tonumber(row[EMB_START_COL + k - 1])
			table.insert(emb, (v and v) or 0)
		end
		table.insert(embList, emb)
	end
	return animIds, clipIds, durations, embList
end

-- ---------------------------------------------------------------------------
-- L2 normalize embeddings in place (per row)
-- ---------------------------------------------------------------------------
local function normalizeEmbeddings(embeddings: { { number } })
	for _, emb in ipairs(embeddings) do
		local norm = 0.0
		for _, x in ipairs(emb) do
			norm += x * x
		end
		norm = math.sqrt(norm)
		if norm < 1e-8 then norm = 1.0 end
		for i = 1, #emb do
			emb[i] = emb[i] / norm
		end
	end
end

-- ---------------------------------------------------------------------------
-- Dot product of two vectors
-- ---------------------------------------------------------------------------
local function dot(a: { number }, b: { number }): number
	local s = 0.0
	for i = 1, math.min(#a, #b) do
		s += a[i] * b[i]
	end
	return s
end

-- ---------------------------------------------------------------------------
-- Union-find: edges = { {i, j}, ... }, n = number of nodes. Returns list of components.
-- ---------------------------------------------------------------------------
local function connectedComponents(n: number, edges: { { number } }): { { number } }
	local parent: { number } = table.create(n)
	for i = 1, n do parent[i] = i end

	local function find(i: number): number
		if parent[i] ~= i then
			parent[i] = find(parent[i])
		end
		return parent[i]
	end

	local function union(i: number, j: number)
		local pi, pj = find(i), find(j)
		if pi ~= pj then
			parent[pi] = pj
		end
	end

	for _, e in ipairs(edges) do
		union(e[1], e[2])
	end

	local compId: { [number]: number } = {}
	local comps: { { number } } = {}
	for i = 1, n do
		local p = find(i)
		if compId[p] == nil then
			table.insert(comps, {})
			compId[p] = #comps
		end
		table.insert(comps[compId[p]], i)
	end
	return comps
end

-- ---------------------------------------------------------------------------
-- R15 character and pose sampling (for detailed comparison)
-- ---------------------------------------------------------------------------
local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "GroupSimilarR15"
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

type AnimTransform = { pos: Vector3, rot: CFrame }
type TrackFrames = { AnimTransform }
type ClipData = { [string]: TrackFrames }

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

local function buildClipData(track: AnimationTrack, duration: number, useFps: number?, useNormDur: number?): (ClipData, number)
	local fps = useFps or FPS
	local normDur = useNormDur or NORMALIZE_DURATION
	local numFrames = math.max(1, math.floor(normDur * fps + 0.5))
	local dt = normDur / math.max(1, numFrames - 1)
	local out: ClipData = {}
	for _, bone in ipairs(R15_BONES) do
		out[bone] = table.create(numFrames)
	end
	for i = 0, numFrames - 1 do
		local srcT = (i * dt) * duration
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
	return out, numFrames
end

local function rotationAngleDeg(cf1: CFrame, cf2: CFrame): number
	local qx1, qy1, qz1, qw1 = cframeToQuat(cf1)
	local qx2, qy2, qz2, qw2 = cframeToQuat(cf2)
	local dot = qx1 * qx2 + qy1 * qy2 + qz1 * qz2 + qw1 * qw2
	if dot < 0 then dot = -dot end
	if dot > 1 then dot = 1 end
	return math.deg(2 * math.acos(dot))
end

local animation = Instance.new("Animation")

local function loadClipAndBuildData(clipPath: string): (ClipData?, number?, number?)
	local instances = FileSystemService:LoadInstances(clipPath)
	local clip = instances and instances[1]
	if not clip then return nil, nil, nil end
	local ok, contentId = pcall(function()
		return KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
	end)
	if not ok or not contentId then
		clip:Destroy()
		return nil, nil, nil
	end
	animation.AnimationId = contentId
	local track = animator:LoadAnimation(animation)
	track:Play(0)
	track.Looped = true
	wait(0)
	local duration = track.Length
	local cs, numFrames = buildClipData(track, duration, VERIFY_FPS, VERIFY_DURATION)
	track:Stop(0)
	track:Destroy()
	clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0)
		t:Destroy()
	end
	animator:StepAnimations(0)
	return cs, numFrames, duration
end

local function compareClipData(csA: ClipData, csB: ClipData, numFrames: number): (number, number)
	local totalPos, totalRot, count = 0.0, 0.0, 0
	for i = 1, numFrames do
		for _, bone in ipairs(R15_BONES) do
			local a = csA[bone] and csA[bone][i]
			local b = csB[bone] and csB[bone][i]
			if a and b then
				totalPos += (a.pos - b.pos).Magnitude
				totalRot += rotationAngleDeg(a.rot, b.rot)
				count += 1
			end
		end
	end
	local n = count
	local meanPos = (n > 0) and (totalPos / n) or 0
	local meanRot = (n > 0) and (totalRot / n) or 0
	return meanPos, meanRot
end

local function duplicateScore(meanPos: number, meanRotDeg: number): number
	local posScore = 1.0 / (1.0 + meanPos * 50)
	local rotScore = 1.0 / (1.0 + meanRotDeg / 10)
	return (posScore + rotScore) / 2
end

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------
print("Loading", INPUT_CSV, "...")
local animIds, clipIds, durations, embeddings = loadFingerprints(INPUT_CSV)
local inputPath = INPUT_CSV
local outputPrefix = OUTPUT_PREFIX
local clipsDir = CLIPS_DIR
if #animIds == 0 then
	print("Trying relative path", INPUT_CSV_REL, "...")
	animIds, clipIds, durations, embeddings = loadFingerprints(INPUT_CSV_REL)
	outputPrefix = OUTPUT_PREFIX_REL
	clipsDir = CLIPS_DIR_REL
end
local N = #animIds
if N == 0 then
	print("No rows loaded.")
	return
end
print("Loaded", N, "animations")

normalizeEmbeddings(embeddings)

-- Pairs above threshold (i < j only); cache embeddings[i] in inner loop
local edges: { { number } } = {}
for i = 1, N do
	local emb_i = embeddings[i]
	for j = i + 1, N do
		if dot(emb_i, embeddings[j]) >= THRESHOLD then
			table.insert(edges, { i, j })
		end
	end
	if i % 500 == 0 then
		print("  similarity pass", i, "/", N)
	end
end
print("Pairs above threshold:", #edges)

-- Detailed verification: only keep edges that pass pose comparison (or skip if SKIP_DETAILED_VERIFY)
local verifiedEdges: { { number } }
if SKIP_DETAILED_VERIFY then
	verifiedEdges = edges
	print("Skipping detailed verification (SKIP_DETAILED_VERIFY=true).")
else
	verifiedEdges = {}
	local uniqueIndices: { [number]: boolean } = {}
	for e = 1, #edges do
		uniqueIndices[edges[e][1]] = true
		uniqueIndices[edges[e][2]] = true
	end
	local numUnique = 0
	for _ in pairs(uniqueIndices) do numUnique += 1 end
	print("  Loading", numUnique, "unique clips (cache) for", #edges, "edge checks...")
	local cache: { [number]: { cs: ClipData, numFrames: number } } = {}
	local loaded = 0
	for idx in pairs(uniqueIndices) do
		local path = clipsDir .. "\\" .. animIds[idx] .. "-" .. clipIds[idx] .. ".rbxm"
		local cs, numFrames = loadClipAndBuildData(path)
		if cs and numFrames then
			cache[idx] = { cs = cs, numFrames = numFrames }
		end
		loaded += 1
		if loaded % 50 == 0 or loaded == numUnique then
			print("    cache load", loaded, "/", numUnique)
		end
	end
	for e = 1, #edges do
		local i, j = edges[e][1], edges[e][2]
		local entryA, entryB = cache[i], cache[j]
		local score = 0.0
		if entryA and entryB and entryA.numFrames == entryB.numFrames then
			local meanPos, meanRot = compareClipData(entryA.cs, entryB.cs, entryA.numFrames)
			score = duplicateScore(meanPos, meanRot)
		end
		if score >= DETAIL_DUP_THRESHOLD then
			table.insert(verifiedEdges, { i, j })
		end
		if e % 500 == 0 or e == #edges then
			print("  detailed check", e, "/", #edges, "-> verified", #verifiedEdges)
		end
	end
	print("Verified edges (actual duplicates):", #verifiedEdges)
end

local comps = connectedComponents(N, verifiedEdges)
-- Sort by size descending
table.sort(comps, function(a, b) return #a > #b end)
local groupsMulti = 0
for _, c in ipairs(comps) do
	if #c > 1 then groupsMulti += 1 end
end
print("Connected components:", #comps, "| Groups with 2+ members:", groupsMulti)

-- animId -> groupId (0-based)
local animToGroup: { [string]: number } = {}
for gid, indices in ipairs(comps) do
	for _, idx in ipairs(indices) do
		animToGroup[animIds[idx]] = gid - 1
	end
end

-- Build CSV rows and sort like anim_dedup: by groupId, then by animId (numeric)
local csvRows: { { string } } = {}
for _, aid in ipairs(animIds) do
	table.insert(csvRows, { aid, tostring(animToGroup[aid]) })
end
local function compareCsvRows(a: { string }, b: { string }): boolean
	local ga, gb = tonumber(a[2]) or 0, tonumber(b[2]) or 0
	if ga ~= gb then return ga < gb end
	return (tonumber(a[1]) or 0) < (tonumber(b[1]) or 0)
end
table.sort(csvRows, compareCsvRows)

local csvPath = outputPrefix .. ".csv"
local csvLines = { "animId,groupId" }
for _, row in ipairs(csvRows) do
	table.insert(csvLines, row[1] .. "," .. row[2])
end
FileSystemService:WriteFile(csvPath, table.concat(csvLines, "\n"), Enum.FileMode.Text)
print("Wrote", csvPath)

local txtPath = outputPrefix .. ".txt"
local txtLines: { string } = {}
for gid, indices in ipairs(comps) do
	if #indices < 2 then continue end
	table.insert(txtLines, "# Group " .. tostring(gid - 1) .. " (" .. tostring(#indices) .. " animations)")
	-- Sort indices by animId for stable output
	table.sort(indices, function(a, b) return animIds[a] < animIds[b] end)
	for _, idx in ipairs(indices) do
		local aid = animIds[idx]
		local dur = durations[aid] or 0
		table.insert(txtLines, "  https://www.roblox.com/catalog/" .. aid .. "  duration=" .. string.format("%.2f", dur) .. "s")
	end
	table.insert(txtLines, "")
end
FileSystemService:WriteFile(txtPath, table.concat(txtLines, "\n"), Enum.FileMode.Text)
print("Wrote", txtPath)
print("Done.")
