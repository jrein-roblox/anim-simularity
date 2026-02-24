--!strict
-- Download animation assets from a CSV file: for each row, load the marketplace
-- asset (Animation in a Model), then load the underlying clip, and write both
-- to out/{id}.rbxm and out/clips/{id}-{clipId}.rbxm.
--
-- Requires: roblox_cli with FStringAuthCookie for InsertService:LoadAsset.
-- Run: roblox-cli run --run <path>/download_assets.lua --fs.readwrite <path> --load.asRobloxScript
--
-- Config: set BASE_PATH and INPUT_CSV below. Ensure out/ and out/clips/ exist.

local FileSystemService = game:GetService("FileSystemService")
local InsertService = game:GetService("InsertService")

-- Config
local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local INPUT_CSV = BASE_PATH .. "\\animations_02-23-26.csv"
local ID_COLUMN = 1   -- 1-based: column index for catalog/asset ID
local SKIP_HEADER = true   -- if true, skip first row (header)
local LOG_EVERY = 100

-- ---------------------------------------------------------------------------
-- CSV parse (handles quoted fields and "" escapes)
-- ---------------------------------------------------------------------------
local function parseCSV(csvData: string, sep: string?, quote: string?): { { string } }
	sep = sep or ","
	quote = quote or '"'
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

local function getAssetIdFromUrl(url: string): number?
	local patterns = {
		"[%?&]id=(%d+)",
		"/(%d+)$",
		"/id/(%d+)",
		"=(%d+)$",
	}
	for _, pattern in ipairs(patterns) do
		local id = string.match(url, pattern)
		if id then return tonumber(id) end
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- Paths
-- ---------------------------------------------------------------------------
local OUT_DIR = BASE_PATH .. "\\out"
local CLIPS_DIR = OUT_DIR .. "\\clips"

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------
local csvData
local ok, err = pcall(function()
	csvData = FileSystemService:ReadFile(INPUT_CSV, Enum.FileMode.Text)
end)
if not ok or not csvData then
	warn("Failed to read CSV:", INPUT_CSV, err or "no data")
	return
end

local rows = parseCSV(csvData)
local startRow = 1
if SKIP_HEADER and #rows > 0 then
	local first = rows[1][ID_COLUMN]
	if not tonumber(first) then
		startRow = 2
	end
end

print("Input CSV:", INPUT_CSV, "| Rows:", #rows - (startRow - 1))
local count = 0
local skipped = 0
local failed = 0

for r = startRow, #rows do
	local row = rows[r]
	if not row or #row < ID_COLUMN then continue end
	local id = tonumber(row[ID_COLUMN])
	if not id then continue end

	local outPath = OUT_DIR .. "\\" .. id .. ".rbxm"
	if FileSystemService:IsRegularFile(outPath) then
		skipped += 1
		count += 1
		if count % LOG_EVERY == 0 then print(count, "skipped (already exists)") end
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

	if not success then
		failed += 1
		warn("Download asset failed:", id, result)
		continue
	end
	if not clipId then
		failed += 1
		warn("No clip ID for asset:", id)
		continue
	end

	local clipPath = CLIPS_DIR .. "\\" .. id .. "-" .. clipId .. ".rbxm"
	if FileSystemService:IsRegularFile(clipPath) then
		count += 1
		if count % LOG_EVERY == 0 then print(count) end
		continue
	end

	success, result = pcall(function()
		local clipModel = InsertService:LoadAsset(clipId)
		if not clipModel then return end
		local clip = clipModel:GetChildren()[1]
		if clip then
			FileSystemService:WriteInstances(clipPath, { clip })
		end
	end)

	if not success then
		failed += 1
		warn("Download clip failed:", id, clipId, result)
		continue
	end

	count += 1
	if count % LOG_EVERY == 0 then print(count) end
end

print("Done. Processed:", count, "| Skipped (exists):", skipped, "| Failed:", failed)
