--!strict
-- Detailed comparison of two animations: when fingerprints are similar, this script
-- samples both at the same normalized timeline and computes per-frame, per-bone pose
-- differences to decide if they are true duplicates (minor edits) vs different animations.
--
-- Input: compare_two.txt in BASE_PATH with one line: animId1,clipId1,animId2,clipId2
--        (animId and clipId from fingerprints.csv; clip files in out/clips/.)
-- Output: compare_report.txt in BASE_PATH with duplicate score (0–1) and per-bone stats.
--
-- Run: roblox-cli run --run <path>/compare_two_anims.lua --fs.readwrite <path> --load.asRobloxScript

local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

RunService:Pause()

local BASE_PATH = "C:\\git\\roblox\\jrein\\anim-simularity"
local CLIPS_DIR = BASE_PATH .. "\\out\\clips"
local COMPARE_INPUT = BASE_PATH .. "\\compare_two.txt"
local COMPARE_REPORT = BASE_PATH .. "\\compare_report.txt"

local FPS = 120
local NORMALIZE_DURATION = 1.0

-- ---------------------------------------------------------------------------
-- R15 character and animator (same as anim_fingerprint)
-- ---------------------------------------------------------------------------
local function spawnR15(cframe: CFrame)
	local desc = Instance.new("HumanoidDescription")
	local character = Players:CreateHumanoidModelFromDescription(desc, Enum.HumanoidRigType.R15)
	character.Name = "CompareR15"
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

-- ---------------------------------------------------------------------------
-- Math: quaternion (same as anim_fingerprint)
-- ---------------------------------------------------------------------------
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

-- Rebuild rotation from quantized quat
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

-- Fix buildClipData to store rotation as CFrame from quantized quat
local function buildClipDataFixed(track: AnimationTrack, duration: number): (ClipData, number)
	local numFrames = math.max(1, math.floor(NORMALIZE_DURATION * FPS + 0.5))
	local dt = NORMALIZE_DURATION / math.max(1, numFrames - 1)
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

-- Angle in degrees between two CFrames (rotation part only)
local function rotationAngleDeg(cf1: CFrame, cf2: CFrame): number
	local qx1, qy1, qz1, qw1 = cframeToQuat(cf1)
	local qx2, qy2, qz2, qw2 = cframeToQuat(cf2)
	local dot = qx1 * qx2 + qy1 * qy2 + qz1 * qz2 + qw1 * qw2
	if dot < 0 then dot = -dot end
	if dot > 1 then dot = 1 end
	local rad = 2 * math.acos(dot)
	return math.deg(rad)
end

-- ---------------------------------------------------------------------------
-- Load one clip and build ClipData
-- ---------------------------------------------------------------------------
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
	local cs, numFrames = buildClipDataFixed(track, duration)
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

-- ---------------------------------------------------------------------------
-- Compare two ClipData (same numFrames and bones)
-- ---------------------------------------------------------------------------
local function compareClipData(csA: ClipData, csB: ClipData, numFrames: number): (number, number, { [string]: { meanPos: number, meanRot: number } })
	local totalPos = 0.0
	local totalRot = 0.0
	local count = 0
	local perBone: { [string]: { meanPos: number, meanRot: number } } = {}
	for _, bone in ipairs(R15_BONES) do
		perBone[bone] = { meanPos = 0, meanRot = 0 }
	end
	for i = 1, numFrames do
		for _, bone in ipairs(R15_BONES) do
			local a = csA[bone] and csA[bone][i]
			local b = csB[bone] and csB[bone][i]
			if a and b then
				local posErr = (a.pos - b.pos).Magnitude
				local rotErr = rotationAngleDeg(a.rot, b.rot)
				totalPos += posErr
				totalRot += rotErr
				count += 1
				perBone[bone].meanPos += posErr
				perBone[bone].meanRot += rotErr
			end
		end
	end
	local n = count
	if n > 0 then
		for _, bone in ipairs(R15_BONES) do
			perBone[bone].meanPos = perBone[bone].meanPos / numFrames
			perBone[bone].meanRot = perBone[bone].meanRot / numFrames
		end
	end
	local meanPos = (n > 0) and (totalPos / n) or 0
	local meanRot = (n > 0) and (totalRot / n) or 0
	return meanPos, meanRot, perBone
end

-- Duplicate score: 1 = identical, 0 = totally different. Use thresholds: pos < 0.01 studs and rot < 2 deg => high score.
local function duplicateScore(meanPos: number, meanRotDeg: number): number
	local posScore = 1.0 / (1.0 + meanPos * 50)   -- 0.02 studs -> ~0.5
	local rotScore = 1.0 / (1.0 + meanRotDeg / 10) -- 10 deg -> ~0.5
	return (posScore + rotScore) / 2
end

-- ---------------------------------------------------------------------------
-- Main: read compare_two.txt, load both clips, compare, write report
-- ---------------------------------------------------------------------------
local inputContent = FileSystemService:ReadFile(COMPARE_INPUT, Enum.FileMode.Text)
local line = (inputContent or ""):gsub("^%s+", ""):gsub("%s+$", "")
local parts = {}
for s in (line .. ","):gmatch("([^,]*),") do
	table.insert(parts, s:gsub("^%s+", ""):gsub("%s+$", ""))
end
if #parts < 4 then
	warn("compare_two.txt must contain one line: animId1,clipId1,animId2,clipId2")
	return
end
local animId1, clipId1, animId2, clipId2 = parts[1], parts[2], parts[3], parts[4]
local path1 = CLIPS_DIR .. "\\" .. animId1 .. "-" .. clipId1 .. ".rbxm"
local path2 = CLIPS_DIR .. "\\" .. animId2 .. "-" .. clipId2 .. ".rbxm"

print("Loading clip A:", path1)
local csA, numFrames, dur1 = loadClipAndBuildData(path1)
if not csA then
	warn("Failed to load clip 1:", path1)
	return
end
print("Loading clip B:", path2)
local csB, numFramesB, dur2 = loadClipAndBuildData(path2)
if not csB then
	warn("Failed to load clip 2:", path2)
	return
end
if numFrames ~= numFramesB then
	warn("Frame count mismatch:", numFrames, "vs", numFramesB)
	return
end

local meanPos, meanRotDeg, perBone = compareClipData(csA, csB, numFrames)
local score = duplicateScore(meanPos, meanRotDeg)

local function verdict(s: number): string
	if s >= 0.9 then return "LIKELY DUPLICATE (minor edits)"
	elseif s >= 0.6 then return "SIMILAR (same motion family)"
	elseif s >= 0.3 then return "SOME SIMILARITY (different animation)"
	else return "DIFFERENT ANIMATION"
	end
end

local report = table.create(0)
table.insert(report, "=== Two-Animation Comparison ===")
table.insert(report, "Anim A: " .. animId1 .. " (" .. string.format("%.2f", dur1) .. "s)  https://www.roblox.com/catalog/" .. animId1)
table.insert(report, "Anim B: " .. animId2 .. " (" .. string.format("%.2f", dur2 or 0) .. "s)  https://www.roblox.com/catalog/" .. animId2)
table.insert(report, "")
table.insert(report, "Normalized timeline: " .. tostring(numFrames) .. " frames @ " .. tostring(NORMALIZE_DURATION) .. "s")
table.insert(report, "Mean position error (studs): " .. string.format("%.6f", meanPos))
table.insert(report, "Mean rotation error (degrees): " .. string.format("%.2f", meanRotDeg))
table.insert(report, "Duplicate score (0-1): " .. string.format("%.4f", score))
table.insert(report, "Verdict: " .. verdict(score))
table.insert(report, "")
table.insert(report, "Per-bone mean position error (studs) / rotation error (deg):")
for _, bone in ipairs(R15_BONES) do
	local b = perBone[bone]
	table.insert(report, "  " .. bone .. ": " .. string.format("%.4f", b.meanPos) .. " / " .. string.format("%.2f", b.meanRot))
end
table.insert(report, "")
table.insert(report, "Done.")

local reportStr = table.concat(report, "\n")
FileSystemService:WriteFile(COMPARE_REPORT, reportStr, Enum.FileMode.Text)
print(reportStr)
print("Wrote", COMPARE_REPORT)
