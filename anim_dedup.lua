-- NOTE: this requires the cli to run with your FStringAuthCookie or it cannot download the assets. You need to build RCC with FStringAuthCookie set in local flags.

-- https://roblox.atlassian.net/wiki/spaces/HOW/pages/1556186296/Roblox+Command+Line+Tool+roblox-cli
-- Run this cmd:
-- C:\PATH\robloxdev-cli.exe run --run C:\PATH\anim-similarity\anim_dedup.lua --fs.readwrite C:\PATH\anim-similarity\ --load.asRobloxScript

--C:\git\roblox\game-engine2\build\ninja\studio\vs2019\x64\optimized\Client\CLI\app\roblox-cli.exe run --run C:\git\roblox\jrein\anim-simularity\anim_dedup.lua --fs.readwrite C:\git\roblox\jrein\anim-simularity\ --load.asRobloxScript 


-- Parse a CSV string into rows of fields ({{string}}).
-- Handles quoted fields (which may span multiple lines) and "" escapes.
local function parseCSV(csvData: string, sep: string?, quote: string?): {{string}}
	sep = sep or ","
	quote = quote or '"'

	-- Normalize line endings to "\n" so we only check one newline code.
	csvData = csvData:gsub("\r\n", "\n"):gsub("\r", "\n")

	local rows = {}
	local row = {}
	local fieldBuf = {}

	local n = #csvData
	local i = 1
	local inQuotes = false

	local function pushField()
		table.insert(row, table.concat(fieldBuf))
		table.clear(fieldBuf)
	end

	local function pushRow()
		-- If the line ends with separators, we still need to push the final empty field.
		pushField()
		table.insert(rows, row)
		row = {}
	end

	while i <= n do
		local c = string.sub(csvData, i, i)

		if inQuotes then
			if c == quote then
				-- Check for escaped quote ("")
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

	-- End of data: commit the last row/field (even if no trailing newline).
	-- If csvData ends with a newline, we pushed the row already; otherwise, finalize here.
	if inQuotes then
		-- Unclosed quote at EOF: take what we have as-is.
		inQuotes = false
	end
	-- If there was content or at least one separator, we should flush.
	if #fieldBuf > 0 or #row > 0 then
		pushRow()
	end

	return rows
end

local function getAssetIdFromUrl(url: string): number?
	-- Try patterns in order of likelihood
	local patterns = {
		"[%?&]id=(%d+)",       -- matches ?id=12345 or &id=12345
		"/(%d+)$",              -- matches ending in /12345
		"/id/(%d+)",            -- matches /id/12345
		"=(%d+)$",              -- matches =12345 at end
	}

	for _, pattern in ipairs(patterns) do
		local id = string.match(url, pattern)
		if id then
			return tonumber(id)
		end
	end

	return nil -- if nothing matched
end

export type AnimTransform = {
	pos: Vector3,     -- local-space
	rot: CFrame,      -- local-space orientation (we'll store quaternion)
}

export type TrackFrames = { AnimTransform }          -- [transform]
export type ClipData = { [string]: TrackFrames }     -- boneName -> TrackFrames

export type FingerprintParams = {
	FPS: number?,                 -- default 60
	DurationMode: "NormalizeTo1" | "Preserve", -- default "NormalizeTo1"
	QuantPos: number?,            -- meters (default 1e-3)
	QuantQuat: number?,           -- quaternion bin (default 1e-4)
	TrackOrder: {string}?,        -- canonical track order; if nil R15
}

local function getCurveTracks(curve: CurveAnimation)
	local tracks = {}

	for _, des in ipairs(curve:GetDescendants()) do
		if des:IsA('Folder') and des:FindFirstChild('Position') and des:FindFirstChild('Rotation') then
			local pos = des:FindFirstChild('Position')
			local rot = des:FindFirstChild('Rotation')
			tracks[des.Name] = {
				pos = pos,
				rot = rot,
			}
		end
	end

	return tracks
end

--CurveAnimation: FloatCurve, Vector3Curve, EulerRotationCurve, RotationCurve, MarkerCurve
local function getCurveValueAtTime(curve: Instance, time: number)
	if curve:IsA('FloatCurve') then
		return curve:GetValueAtTime(time) -- number
	elseif curve:IsA('Vector3Curve') then
		local v = curve:GetValueAtTime(time)
		return Vector3.new(v[1], v[2], v[3]) -- Vector3
	elseif curve:IsA('EulerRotationCurve') then
		return curve:GetRotationAtTime(time) -- CFrame
	elseif curve:IsA('RotationCurve') then
		return curve:GetValueAtTime(time) -- CFrame
	elseif curve:IsA('MarkerCurve') then
		-- NOTES: returns the marker just before or at the time
		local markers = curve:GetMarkers()
		for i=#markers,1,-1 do
			if markers[i].Time <= time then
				return markers[i].Value
			end
		end
		return nil
	else
		warn('Unrecognized curve type')
	end
end

local function calculateCurveAnimLength(curveAnim: Instance): number

	local tracks = getCurveTracks(curveAnim)

	local maxTime = -1
	local function getMaxTimeFromFloatCurveChildren(containerInput: Instance?)
		if not containerInput then
			return
		end
		local container = containerInput :: Instance
		for _, floatCurve in container:GetChildren() do
			if not floatCurve:IsA("FloatCurve") then
				continue
			end
			for _, floatCurveKey in floatCurve:GetKeys() do
				maxTime = math.max(maxTime, floatCurveKey.Time)
			end
		end
	end

	for _, t in tracks do
		getMaxTimeFromFloatCurveChildren(t.pos)
		getMaxTimeFromFloatCurveChildren(t.rot)
	end
	return maxTime
end

-- ==============================
-- Small math helpers (quats etc)
-- ==============================
local function clamp(x, a, b) return math.max(a, math.min(b, x)) end
local function sign(x:number) return (x >= 0) and 1 or -1 end

local function cframeToQuat(cf: CFrame)
	-- Returns normalized quaternion (x,y,z,w) where w is scalar
	local _, _, _, m00,m01,m02, m10,m11,m12, m20,m21,m22 = cf:GetComponents()
	local trace = m00 + m11 + m22
	local x,y,z,w
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
	-- normalize
	local len = math.sqrt(x*x+y*y+z*z+w*w)
	return x/len, y/len, z/len, w/len
end


-- Pack quantized frame into bytes (stable) -----------------
local function quantizeFloat(x:number, q:number)
	return math.floor(x / q + 0.5)
end

local bit32 = bit32
local U32 = 4294967296  -- 2^32

local function u32(n:number): number
	return n % U32
end

-- Safely multiplies two 32-bit integers without exceeding Luau's 53-bit float limit
local function mul32(a: number, b: number): number
	local a_lo = bit32.band(a, 0xFFFF)
	local a_hi = bit32.rshift(a, 16)
	local b_lo = bit32.band(b, 0xFFFF)
	local b_hi = bit32.rshift(b, 16)
	
	-- We ignore a_hi * b_hi because it gets shifted completely out of the 32-bit space
	local mid = (a_hi * b_lo + a_lo * b_hi) % 65536
	local result = (a_lo * b_lo) + (mid * 65536)
	
	return result % U32
end

-- Little-endian 32-bit writer
--local function writeUInt32LE(n:number): string
--	n = u32(n)
--	local b0 = n % 256
--	local b1 = math.floor(n / 256) % 256
--	local b2 = math.floor(n / 65536) % 256
--	local b3 = math.floor(n / 16777216) % 256
--	return string.char(b0, b1, b2, b3)
--end

local function writeUInt32LE(n: number): string
	local b = buffer.create(4)
	buffer.writeu32(b, 0, n)
	return buffer.tostring(b)
end

-- FNV-1a 32 using bxor + mul with wrap
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
	
	for i = 1, bit32.band(len, 0xfffffffc), 4 do
		local k1 = string.unpack("<I4", str, i)
		
		k1 = mul32(k1, c1)
		k1 = bit32.lrotate(k1, 15)
		k1 = mul32(k1, c2)
		
		h1 = bit32.bxor(h1, k1)
		h1 = bit32.lrotate(h1, 13)
		h1 = (mul32(h1, 5) + 0xe6546b64) % U32
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
	h1 = mul32(h1, 0x85ebca6b)
	h1 = bit32.bxor(h1, bit32.rshift(h1, 13))
	h1 = mul32(h1, 0xc2b2ae35)
	h1 = bit32.bxor(h1, bit32.rshift(h1, 16))
	
	return h1
end

local function quatToCFrame(x:number,y:number,z:number,w:number): CFrame
	-- Normalize
	local len = math.sqrt(x*x + y*y + z*z + w*w)
	if len == 0 then return CFrame.new() end
	x,y,z,w = x/len, y/len, z/len, w/len

	-- Convert to rotation matrix
	local xx, yy, zz = x*x, y*y, z*z
	local xy, xz, yz = x*y, x*z, y*z
	local wx, wy, wz = w*x, w*y, w*z

	local m00 = 1 - 2*(yy + zz)
	local m01 = 2*(xy - wz)
	local m02 = 2*(xz + wy)

	local m10 = 2*(xy + wz)
	local m11 = 1 - 2*(xx + zz)
	local m12 = 2*(yz - wx)

	local m20 = 2*(xz - wy)
	local m21 = 2*(yz + wx)
	local m22 = 1 - 2*(xx + yy)

	return CFrame.new(0,0,0, m00,m01,m02, m10,m11,m12, m20,m21,m22)
end


-- Given (curveAnim, boneName, tSeconds), return local-space transform for that bone at t:
local function SampleTrackTransform(track: {}, t: number): AnimTransform
	
	if not track then
		return {
			pos = Vector3.zero,
			rot = CFrame.new(),
		}
	end

	return {
		pos = getCurveValueAtTime(track.pos, t),
		rot = getCurveValueAtTime(track.rot, t),
	}
end


local function buildTrackList(curveAnim: Instance, explicitOrder:{string}?): {string}
	-- supply an explicit canonical list.
	if explicitOrder and #explicitOrder > 0 then
		return table.clone(explicitOrder)
	end

	-- return R15:
	local R15BoneNames = {
		"HumanoidRootPart","LowerTorso","UpperTorso","Head",
		"LeftUpperArm","LeftLowerArm","LeftHand",
		"RightUpperArm","RightLowerArm","RightHand",
		"LeftUpperLeg","LeftLowerLeg","LeftFoot",
		"RightUpperLeg","RightLowerLeg","RightFoot",
	}
	return R15BoneNames
end

local function canonicalize(curveAnim: Instance, duration:number, params: FingerprintParams?): (ClipData, number, {string})
	params = params or {}
	local FPS = params.FPS or 60
	local quantPos = params.QuantPos or 1e-3
	local quantQuat = params.QuantQuat or 1e-4
	local durationMode = params.DurationMode or "NormalizeTo1"
	local framesOverride = params.FramesOverride

	local tracks = buildTrackList(curveAnim, params.TrackOrder)
	local out: ClipData = {}
	
	local tracksCurves = getCurveTracks(curveAnim)

	-- Normalize duration & frame count
	local normDuration = (durationMode == "NormalizeTo1") and 1.0 or duration
	local numFrames = framesOverride or math.max(1, math.floor(normDuration * FPS + 0.5))
	local dt = normDuration / math.max(1, numFrames-1)

	for _, bone in ipairs(tracks) do
		local frames: TrackFrames = table.create(numFrames)
		for i=0,numFrames-1 do
			local t = i * dt
			-- Map resampled time back to source duration if preserving:
			local srcT = (durationMode == "NormalizeTo1") and (t * duration) or t
			
			local trackCurve = tracksCurves[bone]
			local fr = SampleTrackTransform(trackCurve, srcT)

			-- Quantize for stability
			local qx,qy,qz,qw = cframeToQuat(fr.rot)
			qx = quantizeFloat(qx, params.QuantQuat or quantQuat)
			qy = quantizeFloat(qy, params.QuantQuat or quantQuat)
			qz = quantizeFloat(qz, params.QuantQuat or quantQuat)
			qw = quantizeFloat(qw, params.QuantQuat or quantQuat)
			local px = quantizeFloat(fr.pos.X, quantPos)
			local py = quantizeFloat(fr.pos.Y, quantPos)
			local pz = quantizeFloat(fr.pos.Z, quantPos)
			local sx,sy,sz = 1,1,1

			frames[i+1] = {
				pos = Vector3.new(px,py,pz),
				rot = quatToCFrame(qx,qy,qz,qw), 
			}
		end
		out[bone] = frames
	end

	return out, numFrames, tracks
end

-- Serialize canonical clip to bytes in stable order -------------
local function serializeClip(cs: ClipData, tracks:{string}): string
	local chunks = table.create(#tracks * 32)
	for _,bone in ipairs(tracks) do
		local series = cs[bone]
		table.insert(chunks, writeUInt32LE(#series))
		for _,fr in ipairs(series) do
			-- pos
			table.insert(chunks, writeUInt32LE(fr.pos.X))
			table.insert(chunks, writeUInt32LE(fr.pos.Y))
			table.insert(chunks, writeUInt32LE(fr.pos.Z))
			-- rot quaternion
			local qx,qy,qz,qw = cframeToQuat(fr.rot)
			table.insert(chunks, writeUInt32LE(quantizeFloat(qx, 1e-4)))
			table.insert(chunks, writeUInt32LE(quantizeFloat(qy, 1e-4)))
			table.insert(chunks, writeUInt32LE(quantizeFloat(qz, 1e-4)))
			table.insert(chunks, writeUInt32LE(quantizeFloat(qw, 1e-4)))
		end
	end
	return table.concat(chunks)
end


local function hashAnimation(curveAnim: Instance, duration:number, params: FingerprintParams?): number
	local cs, numFrames, tracks = canonicalize(curveAnim, duration, params)
	local bytes = serializeClip(cs, tracks)
	local clipHash = fnv1a32(bytes)
	local clipHash2 = murmur3_32(bytes)
	return clipHash, clipHash2
end


local FileSystemService = game:GetService("FileSystemService")
local InsertService = game:GetService("InsertService")

-- First download all the animations
local download = false
if download then
	local csvData = FileSystemService:ReadFile("C:\\git\\roblox\\jrein\\anim-simularity\\animations_02-05-26.csv", Enum.FileMode.Text)
	local rows = parseCSV(csvData)
	print("Anim count:", #rows)
	for r, row in ipairs(rows) do
		--print(("Row %d: (%d fields)"):format(r, #row))
		--print("  id:", row[1], "name:", row[4])		
		local id = tonumber(row[1])
		
		local fileName = "C:\\git\\roblox\\jrein\\anim-simularity\\out\\" .. id .. ".rbxm"
		print(fileName)
		if FileSystemService:IsRegularFile(fileName) then
			--print("Already exists!")
			continue
		end
		
		local clipId
		local success, result = pcall(function()
			local model = InsertService:LoadAsset(id)
			if model then
				local animation = model:GetChildren()[1]
				print("Clip", animation.AnimationId)
				clipId = getAssetIdFromUrl(animation.AnimationId)
				FileSystemService:WriteInstances(fileName, {animation})
			end
		end)
		
		if success == false then
			print("Error Downloading Animation:", result)
			continue
		end
		
		local clipFileName = "C:\\git\\roblox\\jrein\\anim-simularity\\out\\clips\\" .. id .. "-" .. clipId .. ".rbxm"	
		
		local success, result = pcall(function()
			local clip = InsertService:LoadAsset(clipId)
			print(clip)
			if clip then
				FileSystemService:WriteInstances(clipFileName, {clip:GetChildren()[1]})
			end
		end)
		
		if success == false then
			print("Error Downloading Clip:", result)
			continue
		end
	end
end

-- next build hashes
local hashmap = {}
local hashmap2 = {}
local hashmap3 = {}
local csv = ""--"animId,clipId,duration,hash\n"
local count = 0
for fileData in FileSystemService:Walk("C:\\git\\roblox\\jrein\\anim-simularity\\out\\clips\\", Enum.FileSystemWalkMode.NonRecursive) do
	--print(fileData.Path)
	local clip = FileSystemService:LoadInstances(fileData.Path)[1]
	
	-- TODO: support KFS
	if not clip or not clip:IsA("CurveAnimation") then
		--print("clip is not a curve animation", clip, clip.ClassName, fileData.Path)
		continue
	end
	
	local animId, clipId = fileData.Path:match(".*/(%d+)%-(%d+)%.rbxm$")

	local line = ""
	local success, result = pcall(function()
		local duration = calculateCurveAnimLength(clip)
		local clipHash, clipHash2 = hashAnimation(clip, duration, { FPS = 120, DurationMode = "NormalizeTo1" })
		
		--if hashmap[clipHash] == nil then
		--	hashmap[clipHash] = {animId}
		--else
		--	table.insert(hashmap[clipHash], animId)
		--end
		--
		--if hashmap2[clipHash2] == nil then
		--	hashmap2[clipHash2] = {animId}
		--else
		--	table.insert(hashmap2[clipHash2], animId)
		--end
		
		local clipHash3 = string.format("%08X%08X", clipHash, clipHash2)
		if hashmap3[clipHash3] == nil then
			hashmap3[clipHash3] = {animId}
		else
			table.insert(hashmap3[clipHash3], animId)
		end
		
		line = animId .. "," .. clipId .. "," .. duration .. "," .. clipHash .. "," .. clipHash2 ",\n"
		csv = csv .. line
	end)
	
	count = count + 1
	
	if count % 100 == 0 then
		print(count)
		--print(line)
	end
	
	--if count > 2000 then
	--	break
	--end
	
	clip:Destroy()
end

FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\hashes.csv", csv, Enum.FileMode.Text)

function compare_numbers(a, b)
    return tonumber(a) < tonumber(b)
end

function compare_tables(a, b)
    return tonumber(a[1]) < tonumber(b[1])
end

--local sortedTable = {}
--for key, value in pairs(hashmap) do
--	table.sort(value, compare_numbers)
--	table.insert(sortedTable, value)
--end
--table.sort(sortedTable, compare_tables)
--
--local sortedTable2 = {}
--for key, value in pairs(hashmap2) do
--	table.sort(value, compare_numbers)
--	table.insert(sortedTable2, value)
--end
--table.sort(sortedTable2, compare_tables)

local sortedTable3 = {}
for key, value in pairs(hashmap3) do
	table.sort(value, compare_numbers)
	table.insert(sortedTable3, value)
end
table.sort(sortedTable3, compare_tables)

--local out = ""
--for i, value in ipairs(sortedTable) do
--	if #value > 1 then
--		local line = ""
--		for i, v in ipairs(value) do
--			line = line .. v .. ","
--		end
--		out = out .. line .. "\n"
--	end
--end
--
----print(out)
--FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\dupes.csv", out, Enum.FileMode.Text)
--
--
--out = ""
--for i, value in ipairs(sortedTable2) do
--	if #value > 1 then
--		local line = ""
--		for i, v in ipairs(value) do
--			line = line .. v .. ","
--		end
--		out = out .. line .. "\n"
--	end
--end
--
----print(out)
--FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\dupes2.csv", out, Enum.FileMode.Text)

out = ""
for i, value in ipairs(sortedTable3) do
	if #value > 1 then
		local line = ""
		for i, v in ipairs(value) do
			line = line .. v .. ","
		end
		out = out .. line .. "\n"
	end
end

--print(out)
FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\dupes3.csv", out, Enum.FileMode.Text)