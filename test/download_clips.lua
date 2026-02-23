-- NOTE: this requires the cli to run with your FStringAuthCookie or it cannot download the assets

-- https://roblox.atlassian.net/wiki/spaces/HOW/pages/1556186296/Roblox+Command+Line+Tool+roblox-cli
-- Run this cmd:
-- C:\PATH\robloxdev-cli.exe run --run C:\PATH\anim-similarity\download_clips.lua --fs.readwrite C:\PATH\anim-similarity\ --load.asRobloxScript

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


local FileSystemService = game:GetService("FileSystemService")
local InsertService = game:GetService("InsertService")
local csvData = FileSystemService:ReadFile("C:\\git\\roblox\\anim-similarity\\animations.csv", Enum.FileMode.Text)
local rows = parseCSV(csvData)

print("Row count:", #rows)
for r, row in ipairs(rows) do
	print(("Row %d: (%d fields)"):format(r, #row))
	print("  id:", row[1], "name:", row[4])

	
	local id = tonumber(row[1])
	local fileName = "C:\\git\\roblox\\anim-similarity\\out\\" .. id .. ".rbxm"	
	
	if FileSystemService:IsRegularFile(fileName) then
		print("Already exists!")
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
	
	local clipFileName = "C:\\git\\roblox\\anim-similarity\\out\\clips\\" .. id .. "-" .. clipId .. ".rbxm"	
	
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
