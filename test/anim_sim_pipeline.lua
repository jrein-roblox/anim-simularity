--!strict
-- Combined pipeline: download assets | fingerprint clips | group similar.
-- Mode and options are driven by command-line arguments (after --) via ProcessService:GetCommandLineArgs().
--
-- Run: roblox-cli run --run <path>/anim_sim_pipeline.lua --fs.readwrite <path> -- --mode fingerprint --base .
--
-- Arguments (after the double dash --):
--   --mode download|fingerprint|group   (required)
--   --base <path>   Base path; default .
--   --input <path>  Input CSV (download: animations list; group: fingerprints.csv)
--   --idColumn <n>  --skipHeader true|false  --logEvery <n>   (download)
--   --fps <n>  --normDur <n>  --flushEvery <n>   (fingerprint)
--   --threshold <n>  --detailThreshold <n>  --skipDetailVerify  --verifyFps <n>  --verifyDur <n>   (group)
--
-- Modes:
--   download   Input CSV -> out/{id}.rbxm, out/clips/{id}-{clipId}.rbxm
--   fingerprint  Walk out/clips -> fingerprints.csv
--   group     fingerprints.csv -> groups.csv, groups.txt

local FileSystemService = game:GetService("FileSystemService")
local InsertService = game:GetService("InsertService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local ProcessService = game:GetService("ProcessService")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

-- ---------------------------------------------------------------------------
-- Arguments: from command line (after --) via ProcessService:GetCommandLineArgs()
-- Format: --key value  or  --key=value  (flag with no value -> "true")
-- ---------------------------------------------------------------------------
local function getArgsFromCommandLine(): { [string]: string }
	local out: { [string]: string } = {}
	local ok, raw = pcall(function()
		return ProcessService:GetCommandLineArgs()
	end)
	if not ok or not raw or type(raw) ~= "table" then return out end
	local i = 1
	while i <= #raw do
		local a = raw[i]
		if type(a) == "string" and a:sub(1, 2) == "--" then
			local key = a:sub(3)
			local val: string? = nil
			local eq = key:find("=")
			if eq then
				val = key:sub(eq + 1)
				key = key:sub(1, eq - 1)
			end
			if val == nil and i + 1 <= #raw then
				local nextArg = raw[i + 1]
				if type(nextArg) == "string" and nextArg:sub(1, 2) ~= "--" then
					i += 1
					val = nextArg
				end
			end
			if key and key ~= "" then
				out[key] = (val == nil or val == "") and "true" or val
			end
		end
		i += 1
	end
	return out
end

local function getArgs(): { [string]: string }
	-- Command line (after --) is primary
	local out = getArgsFromCommandLine()
	-- Optional: overlay from args file for any missing keys (defaults)
	local tryPaths = { "anim_sim_args.txt" }
	if out["base"] and out["base"] ~= "." and out["base"] ~= "" then
		table.insert(tryPaths, 1, out["base"] .. "\\anim_sim_args.txt")
	end
	for _, p in ipairs(tryPaths) do
		local ok, content = pcall(function()
			return FileSystemService:ReadFile(p, Enum.FileMode.Text)
		end)
		if ok and content and #content > 0 then
			for line in (content .. "\n"):gmatch("([^\r\n]+)") do
				local key, val = line:match("^%s*([%w_]+)%s*=%s*(.*)%s*$")
				if key and val and out[key] == nil then
					out[key] = val:gsub("^%s+"):gsub("%s+$")
				end
			end
			break
		end
	end
	return out
end

local function getArg(args: { [string]: string }, key: string, default: string): string
	local v = args[key]
	if v ~= nil and v ~= "" then return v end
	return default
end

local function getArgNum(args: { [string]: string }, key: string, default: number): number
	local v = args[key]
	if v == nil or v == "" then return default end
	return tonumber(v) or default
end

local function getArgBool(args: { [string]: string }, key: string, default: boolean): boolean
	local v = args[key]
	if v == nil or v == "" then return default end
	local lower = (v :: string):lower()
	if lower == "true" or lower == "1" or lower == "yes" then return true end
	if lower == "false" or lower == "0" or lower == "no" then return false end
	return default
end

-- ---------------------------------------------------------------------------
-- Shared: CSV parse
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
			if c == quote then inQuotes = true; i += 1
			elseif c == sep then pushField(); i += 1
			elseif c == "\n" then pushRow(); i += 1
			else table.insert(fieldBuf, c); i += 1
			end
		end
	end
	if #fieldBuf > 0 or #row > 0 then pushRow() end
	return rows
end

-- ---------------------------------------------------------------------------
-- Mode: download
-- ---------------------------------------------------------------------------
local function getAssetIdFromUrl(url: string): number?
	local patterns = { "[%?&]id=(%d+)", "/(%d+)$", "/id/(%d+)", "=(%d+)$" }
	for _, pattern in ipairs(patterns) do
		local id = string.match(url, pattern)
		if id then return tonumber(id) end
	end
	return nil
end

local function runDownload(args: { [string]: string })
	local base = getArg(args, "base", ".")
	local input = getArg(args, "input", "animations.csv")
	local idCol = getArgNum(args, "idColumn", 1)
	local skipHeader = getArgBool(args, "skipHeader", true)
	local logEvery = getArgNum(args, "logEvery", 100)
	local inputPath = (base == "." or base == "") and input or (base .. "\\" .. input)
	local outDir = (base == "." or base == "") and "out" or (base .. "\\out")
	local clipsDir = outDir .. "\\clips"

	local ok, csvData = pcall(function()
		return FileSystemService:ReadFile(inputPath, Enum.FileMode.Text)
	end)
	if not ok or not csvData then
		warn("Download: Failed to read", inputPath)
		return
	end
	local rows = parseCSV(csvData)
	local startRow = 1
	if skipHeader and #rows > 0 and not tonumber(rows[1][idCol]) then
		startRow = 2
	end
	print("Download: Input", inputPath, "| Rows:", #rows - startRow + 1)
	local count, skipped, failed = 0, 0, 0
	for r = startRow, #rows do
		local row = rows[r]
		if not row or #row < idCol then continue end
		local id = tonumber(row[idCol])
		if not id then continue end
		local outPath = outDir .. "\\" .. id .. ".rbxm"
		if FileSystemService:IsRegularFile(outPath) then
			skipped += 1
			count += 1
			if count % logEvery == 0 then print(count, "skipped (exists)") end
			continue
		end
		local clipId: number? = nil
		local success, result = pcall(function()
			local model = InsertService:LoadAsset(id)
			if not model then return end
			local animation = model:GetChildren()[1]
			if not animation then return end
			clipId = getAssetIdFromUrl(animation.AnimationId)
			FileSystemService:WriteInstances(outPath, { animation })
		end)
		if not success then failed += 1; warn("Download asset failed:", id, result); continue end
		if not clipId then failed += 1; warn("No clip ID for asset:", id); continue end
		local clipPath = clipsDir .. "\\" .. id .. "-" .. clipId .. ".rbxm"
		if FileSystemService:IsRegularFile(clipPath) then
			count += 1
			if count % logEvery == 0 then print(count) end
			continue
		end
		success, result = pcall(function()
			local clipModel = InsertService:LoadAsset(clipId)
			if not clipModel then return end
			local clip = clipModel:GetChildren()[1]
			if clip then FileSystemService:WriteInstances(clipPath, { clip }) end
		end)
		if not success then failed += 1; warn("Download clip failed:", id, clipId, result); continue end
		count += 1
		if count % logEvery == 0 then print(count) end
	end
	print("Download done. Processed:", count, "| Skipped:", skipped, "| Failed:", failed)
end

-- ---------------------------------------------------------------------------
-- Shared: R15 character and pose (for fingerprint + group verify)
-- ---------------------------------------------------------------------------
local R15_BONES = {
	"LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}

type AnimTransform = { pos: Vector3, rot: CFrame }
type ClipData = { [string]: { AnimTransform } }

local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "AnimSimR15"
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

local quantPos = 1e-3
local quantQuat = 1e-4
local function quantizeFloat(x: number, q: number): number return math.floor(x / q + 0.5) end

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

local function buildClipData(track: AnimationTrack, duration: number, fps: number, normDur: number): (ClipData, number)
	local numFrames = math.max(1, math.floor(normDur * fps + 0.5))
	local dt = normDur / math.max(1, numFrames - 1)
	local out: ClipData = {}
	for _, bone in ipairs(R15_BONES) do out[bone] = table.create(numFrames) end
	for i = 0, numFrames - 1 do
		track.TimePosition = (i * dt) * duration
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
	return out, numFrames
end

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

local function buildEmbedding(cs: ClipData, tracks: { string }): { number }
	local feat: { number } = {}
	local function push(x: number) table.insert(feat, x) end
	local function push3(x, y, z) table.insert(feat, x); table.insert(feat, y); table.insert(feat, z) end
	for _, bone in ipairs(tracks) do
		local s = cs[bone]
		local n = s and #s or 0
		if n == 0 then for _ = 1, 16 do push(0.0) end
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
					velEX += math.abs(p.X - prev.X); velEY += math.abs(p.Y - prev.Y); velEZ += math.abs(p.Z - prev.Z)
					accVar += (p.X - prev.X) ^ 2 + (p.Y - prev.Y) ^ 2 + (p.Z - prev.Z) ^ 2
					prev = p
				end
				local invN1 = 1.0 / (n - 1)
				velEX *= invN1; velEY *= invN1; velEZ *= invN1
			end
			push3(meanPX, meanPY, meanPZ)
			push3(meanLX, meanLY, meanLZ)
			local varPos = math.sqrt(varPX + varPY + varPZ)
			local varRot = math.sqrt(varLX + varLY + varLZ)
			local velMag = velEX + velEY + velEZ
			local still = 1.0 / (1.0 + accVar)
			table.insert(feat, varPos + varRot + velMag * 0.1 + still * 0.1)
		end
	end
	while #feat < 128 do table.insert(feat, 0.0) end
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

-- ---------------------------------------------------------------------------
-- Mode: fingerprint
-- ---------------------------------------------------------------------------
local animation = Instance.new("Animation")
local function runFingerprint(args: { [string]: string })
	local base = getArg(args, "base", ".")
	local fps = getArgNum(args, "fps", 120)
	local normDur = getArgNum(args, "normDur", 1)
	local clipsDir = (base == "." or base == "") and "out\\clips" or (base .. "\\out\\clips")
	local outputCsv = (base == "." or base == "") and "fingerprints.csv" or (base .. "\\fingerprints.csv")
	local flushEvery = getArgNum(args, "flushEvery", 2000)

	local csv = "animId,clipId,duration,hash"
	for i = 1, 128 do csv = csv .. ",emb" .. tostring(i) end
	csv = csv .. "\n"
	local count = 0
	for fileData in FileSystemService:Walk(clipsDir, Enum.FileSystemWalkMode.NonRecursive) do
		local path = fileData.Path
		local instances = FileSystemService:LoadInstances(path)
		local clip = instances and instances[1]
		if not clip then continue end
		local ok, contentId = pcall(function()
			return KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
		end)
		if not ok or not contentId then clip:Destroy(); continue end
		animation.AnimationId = contentId
		local track = animator:LoadAnimation(animation)
		track:Play(0)
		track.Looped = true
		wait(0)
		local duration = track.Length
		local animId, clipId = path:match("(%d+)%-(%d+)%.rbxm$")
		if not animId or not clipId then animId = "?"; clipId = "?" end
		local line = ""
		local success, err = pcall(function()
			local cs, numFrames = buildClipData(track, duration, fps, normDur)
			local embedding = buildEmbedding(cs, R15_BONES)
			line = animId .. "," .. clipId .. "," .. string.format("%.6f", duration) .. ",0"
			for _, v in ipairs(embedding) do line = line .. "," .. string.format("%.8f", v) end
			line = line .. "\n"
		end)
		if success and #line > 0 then
			csv = csv .. line
		elseif not success then
			warn("Fingerprint failed:", path, err)
		end
		track:Stop(0); track:Destroy(); clip:Destroy()
		for _, t in ipairs(animator:GetPlayingAnimationTracks()) do t:Stop(0); t:Destroy() end
		animator:StepAnimations(0)
		count += 1
		if count % 100 == 0 then print(count) end
		if count % flushEvery == 0 and #csv > 0 then
			FileSystemService:WriteFile(outputCsv, csv, Enum.FileMode.Text)
		end
	end
	if #csv > 0 then FileSystemService:WriteFile(outputCsv, csv, Enum.FileMode.Text) end
	print("Fingerprint done. Wrote", count, "rows to", outputCsv)
end

-- ---------------------------------------------------------------------------
-- Mode: group (similarity + optional detail verify)
-- ---------------------------------------------------------------------------
local EMB_DIM = 128
local EMB_START_COL = 5

local function loadFingerprints(path: string): ({ string }, { string }, { [string]: number }, { { number } })
	local animIds, clipIds = {}, {}
	local durations: { [string]: number } = {}
	local embList: { { number } } = {}
	local ok, csvData = pcall(function() return FileSystemService:ReadFile(path, Enum.FileMode.Text) end)
	if not ok or not csvData then return animIds, clipIds, durations, embList end
	local rows = parseCSV(csvData)
	local startRow = (#rows >= 1 and not tonumber(rows[1][1])) and 2 or 1
	for r = startRow, #rows do
		local row = rows[r]
		if not row or #row < EMB_START_COL + EMB_DIM - 1 then continue end
		table.insert(animIds, row[1])
		table.insert(clipIds, row[2] or "")
		durations[row[1]] = tonumber(row[3]) or 0
		local emb: { number } = {}
		for k = 1, EMB_DIM do table.insert(emb, tonumber(row[EMB_START_COL + k - 1]) or 0) end
		table.insert(embList, emb)
	end
	return animIds, clipIds, durations, embList
end

local function normalizeEmbeddings(embeddings: { { number } })
	for _, emb in ipairs(embeddings) do
		local norm = 0.0
		for _, x in ipairs(emb) do norm += x * x end
		norm = math.sqrt(norm); if norm < 1e-8 then norm = 1.0 end
		for i = 1, #emb do emb[i] = emb[i] / norm end
	end
end

local function dot(a: { number }, b: { number }): number
	local s = 0.0
	for i = 1, math.min(#a, #b) do s += a[i] * b[i] end
	return s
end

local function connectedComponents(n: number, edges: { { number } }): { { number } }
	local parent = table.create(n)
	for i = 1, n do parent[i] = i end
	local function find(i: number): number
		if parent[i] ~= i then parent[i] = find(parent[i]) end
		return parent[i]
	end
	local function union(i, j)
		local pi, pj = find(i), find(j)
		if pi ~= pj then parent[pi] = pj end
	end
	for _, e in ipairs(edges) do union(e[1], e[2]) end
	local compId: { [number]: number } = {}
	local comps: { { number } } = {}
	for i = 1, n do
		local p = find(i)
		if compId[p] == nil then table.insert(comps, {}); compId[p] = #comps end
		table.insert(comps[compId[p]], i)
	end
	return comps
end

local function rotationAngleDeg(cf1: CFrame, cf2: CFrame): number
	local qx1, qy1, qz1, qw1 = cframeToQuat(cf1)
	local qx2, qy2, qz2, qw2 = cframeToQuat(cf2)
	local d = qx1 * qx2 + qy1 * qy2 + qz1 * qz2 + qw1 * qw2
	if d < 0 then d = -d end
	if d > 1 then d = 1 end
	return math.deg(2 * math.acos(d))
end

local function loadClipAndBuildData(clipPath: string, fps: number, normDur: number): (ClipData?, number?)
	local instances = FileSystemService:LoadInstances(clipPath)
	local clip = instances and instances[1]
	if not clip then return nil, nil end
	local ok, contentId = pcall(function() return KeyframeSequenceProvider:RegisterKeyframeSequence(clip) end)
	if not ok or not contentId then clip:Destroy(); return nil, nil end
	animation.AnimationId = contentId
	local track = animator:LoadAnimation(animation)
	track:Play(0); track.Looped = true; wait(0)
	local duration = track.Length
	local cs, numFrames = buildClipData(track, duration, fps, normDur)
	track:Stop(0); track:Destroy(); clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do t:Stop(0); t:Destroy() end
	animator:StepAnimations(0)
	return cs, numFrames
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
	return (n > 0) and (totalPos / n) or 0, (n > 0) and (totalRot / n) or 0
end

local function duplicateScore(meanPos: number, meanRotDeg: number): number
	return ((1.0 / (1.0 + meanPos * 50)) + (1.0 / (1.0 + meanRotDeg / 10))) / 2
end

local function runGroup(args: { [string]: string })
	local base = getArg(args, "base", ".")
	local inputCsv = getArg(args, "input", "fingerprints.csv")
	local clipsDir = (base == "." or base == "") and "out\\clips" or (base .. "\\out\\clips")
	local outputPrefix = (base == "." or base == "") and "groups" or (base .. "\\groups")
	local threshold = getArgNum(args, "threshold", 0.99)
	local detailThreshold = getArgNum(args, "detailThreshold", 0.7)
	local skipVerify = getArgBool(args, "skipDetailVerify", false)
	local verifyFps = getArgNum(args, "verifyFps", 30)
	local verifyDur = getArgNum(args, "verifyDur", 1)

	local path = (base == "." or base == "") and inputCsv or (base .. "\\" .. inputCsv)
	print("Group: Loading", path, "...")
	local animIds, clipIds, durations, embeddings = loadFingerprints(path)
	if #animIds == 0 then
		path = inputCsv
		animIds, clipIds, durations, embeddings = loadFingerprints(path)
	end
	local N = #animIds
	if N == 0 then print("No rows loaded."); return end
	print("Loaded", N, "animations")
	normalizeEmbeddings(embeddings)

	local edges: { { number } } = {}
	for i = 1, N do
		local emb_i = embeddings[i]
		for j = i + 1, N do
			if dot(emb_i, embeddings[j]) >= threshold then
				table.insert(edges, { i, j })
			end
		end
		if i % 500 == 0 then print("  similarity", i, "/", N) end
	end
	print("Pairs above threshold:", #edges)

	local verifiedEdges: { { number } }
	if skipVerify then
		verifiedEdges = edges
		print("Skipping detailed verification.")
	else
		verifiedEdges = {}
		local uniqueIndices: { [number]: boolean } = {}
		for e = 1, #edges do uniqueIndices[edges[e][1]] = true; uniqueIndices[edges[e][2]] = true end
		local numUnique = 0
		for _ in pairs(uniqueIndices) do numUnique += 1 end
		print("  Loading", numUnique, "unique clips for", #edges, "edge checks...")
		local cache: { [number]: { cs: ClipData, numFrames: number } } = {}
		local loaded = 0
		for idx in pairs(uniqueIndices) do
			local clipPath = clipsDir .. "\\" .. animIds[idx] .. "-" .. clipIds[idx] .. ".rbxm"
			local cs, numFrames = loadClipAndBuildData(clipPath, verifyFps, verifyDur)
			if cs and numFrames then cache[idx] = { cs = cs, numFrames = numFrames } end
			loaded += 1
			if loaded % 50 == 0 or loaded == numUnique then print("    cache", loaded, "/", numUnique) end
		end
		for e = 1, #edges do
			local i, j = edges[e][1], edges[e][2]
			local entryA, entryB = cache[i], cache[j]
			local score = 0.0
			if entryA and entryB and entryA.numFrames == entryB.numFrames then
				local meanPos, meanRot = compareClipData(entryA.cs, entryB.cs, entryA.numFrames)
				score = duplicateScore(meanPos, meanRot)
			end
			if score >= detailThreshold then table.insert(verifiedEdges, { i, j }) end
			if e % 500 == 0 or e == #edges then print("  verify", e, "/", #edges, "->", #verifiedEdges) end
		end
		print("Verified edges:", #verifiedEdges)
	end

	local comps = connectedComponents(N, verifiedEdges)
	table.sort(comps, function(a, b) return #a > #b end)
	local animToGroup: { [string]: number } = {}
	for gid, indices in ipairs(comps) do
		for _, idx in ipairs(indices) do animToGroup[animIds[idx]] = gid - 1 end
	end
	local csvRows: { { string } } = {}
	for _, aid in ipairs(animIds) do table.insert(csvRows, { aid, tostring(animToGroup[aid]) }) end
	table.sort(csvRows, function(a, b)
		local ga, gb = tonumber(a[2]) or 0, tonumber(b[2]) or 0
		if ga ~= gb then return ga < gb end
		return (tonumber(a[1]) or 0) < (tonumber(b[1]) or 0)
	end)
	local csvLines = { "animId,groupId" }
	for _, row in ipairs(csvRows) do table.insert(csvLines, row[1] .. "," .. row[2]) end
	FileSystemService:WriteFile(outputPrefix .. ".csv", table.concat(csvLines, "\n"), Enum.FileMode.Text)
	print("Wrote", outputPrefix .. ".csv")
	local txtLines: { string } = {}
	for gid, indices in ipairs(comps) do
		if #indices < 2 then continue end
		table.insert(txtLines, "# Group " .. tostring(gid - 1) .. " (" .. tostring(#indices) .. " animations)")
		table.sort(indices, function(a, b) return animIds[a] < animIds[b] end)
		for _, idx in ipairs(indices) do
			table.insert(txtLines, "  https://www.roblox.com/catalog/" .. animIds[idx] .. "  duration=" .. string.format("%.2f", durations[animIds[idx]] or 0) .. "s")
		end
		table.insert(txtLines, "")
	end
	FileSystemService:WriteFile(outputPrefix .. ".txt", table.concat(txtLines, "\n"), Enum.FileMode.Text)
	print("Wrote", outputPrefix .. ".txt")
	print("Group done.")
end

-- ---------------------------------------------------------------------------
-- Main dispatch
-- ---------------------------------------------------------------------------
local args = getArgs()
local mode = (getArg(args, "mode", "")):lower()
if mode == "download" then
	runDownload(args)
elseif mode == "fingerprint" then
	runFingerprint(args)
elseif mode == "group" then
	runGroup(args)
else
	print("anim_sim_pipeline.lua: pass arguments after the double dash --")
	print("  roblox-cli run --run anim_sim_pipeline.lua --fs.readwrite <path> -- --mode <mode> [options]")
	print("  Example: -- --mode fingerprint --base .")
	print("  Modes: download | fingerprint | group")
	print("  Options: --base, --input, --idColumn, --skipHeader, --fps, --normDur, --threshold, --detailThreshold, --skipDetailVerify, --verifyFps, --verifyDur, --flushEvery, --logEvery")
end
