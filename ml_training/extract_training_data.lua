--!strict
-- Extract full-clip pose data at 30 FPS for ML training.
-- Output: one CSV per clip under train_data/ (variable-length rows), plus manifest.csv.
-- Run: roblox-cli run --run ml_training/extract_training_data.lua --fs.readwrite <repo> --load.asRobloxScript

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

local CLIPS_DIR = "out/clips"
local TRAIN_DATA_DIR = "ml_training/train_data"
local FPS = 30

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

local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "AnimExtractR15"
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
local animator = character.Humanoid.Animator
local hrp = character:FindFirstChild("HumanoidRootPart") :: BasePart

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

-- Sample every frame at FPS for full clip duration (no fixed cap).
local function buildClipDataFull(track: AnimationTrack, duration: number): (ClipData, number, { string })
	local numFrames = math.max(1, math.ceil(duration * FPS))
	local out: ClipData = {}
	for _, bone in ipairs(R15_BONES) do
		out[bone] = table.create(numFrames)
	end
	for i = 0, numFrames - 1 do
		local t = (i + 0.5) / FPS
		if t > duration then break end
		local srcT = math.min(t, duration - 1e-6)
		track.TimePosition = srcT
		animator:StepAnimations(0)
		local pose = sampleRootSpacePose()
		for _, bone in ipairs(R15_BONES) do
			out[bone][i + 1] = pose[bone]
		end
	end
	return out, numFrames, R15_BONES
end

-- Coerce to finite number so we never write -nan(ind) or inf to CSV
local function safeNum(v: number): number
	if v ~= v or v == math.huge or v == -math.huge then
		return 0
	end
	return v
end

-- Build header: frame, then for each bone: px,py,pz,lx,ly,lz
local function csvHeader(): string
	local parts = { "frame" }
	for _, bone in ipairs(R15_BONES) do
		table.insert(parts, bone .. "_px")
		table.insert(parts, bone .. "_py")
		table.insert(parts, bone .. "_pz")
		table.insert(parts, bone .. "_lx")
		table.insert(parts, bone .. "_ly")
		table.insert(parts, bone .. "_lz")
	end
	return table.concat(parts, ",")
end

-- Write one clip CSV and return numFrames.
local function writeClipCSV(cs: ClipData, numFrames: number, tracks: { string }, filePath: string): number
	local lines = { csvHeader() }
	local qx0, qy0, qz0, qw0 = cframeToQuat(cs[tracks[1]][1].rot)
	for i = 1, numFrames - 1 do
		local row = { tostring(i - 1) }
		for _, bone in ipairs(tracks) do
			local fr = cs[bone][i]
			local px, py, pz = fr.pos.X, fr.pos.Y, fr.pos.Z
			local qx, qy, qz, qw = cframeToQuat(fr.rot)
			qx, qy, qz, qw = quatHemisphereAlign(qx0, qy0, qz0, qw0, qx, qy, qz, qw)
			local lx, ly, lz = quatLog(qx, qy, qz, qw)
			table.insert(row, string.format("%.8f", safeNum(px)))
			table.insert(row, string.format("%.8f", safeNum(py)))
			table.insert(row, string.format("%.8f", safeNum(pz)))
			table.insert(row, string.format("%.8f", safeNum(lx)))
			table.insert(row, string.format("%.8f", safeNum(ly)))
			table.insert(row, string.format("%.8f", safeNum(lz)))
		end
		table.insert(lines, table.concat(row, ","))
	end
	FileSystemService:WriteFile(filePath, table.concat(lines, "\n"), Enum.FileMode.Text)
	return numFrames
end

local animation = Instance.new("Animation")
local manifestRows = { "animId,clipId,duration,numFrames" }
local count = 0

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
		local cs, numFrames, tracks = buildClipDataFull(track, duration)
		local outPath = TRAIN_DATA_DIR .. "/" .. animId .. "-" .. clipId .. ".csv"
		writeClipCSV(cs, numFrames, tracks, outPath)
		table.insert(manifestRows, string.format("%s,%s,%.6f,%d", animId, clipId, duration, numFrames))
	end)

	if not success then
		warn("Extract failed:", path, err)
	end

	count = count + 1
	if count % 50 == 0 then print(count) end

	track:Stop(0)
	track:Destroy()
	clip:Destroy()
	for _, t in ipairs(animator:GetPlayingAnimationTracks()) do
		t:Stop(0)
		t:Destroy()
	end
	animator:StepAnimations(0)
end

FileSystemService:WriteFile(TRAIN_DATA_DIR .. "/manifest.csv", table.concat(manifestRows, "\n"), Enum.FileMode.Text)
print("Done. Wrote", count, "clips to", TRAIN_DATA_DIR)
