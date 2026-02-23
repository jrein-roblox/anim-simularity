--"run --run C:\\git\\roblox\\jrein\\anim-simularity\\test.lua --fs.readwrite C:\\git\\roblox\\jrein\\anim-simularity\\ --load.asRobloxScript"


print("Running:")
local FileSystemService = game:GetService("FileSystemService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local RunService = game:GetService("RunService")
RunService:Pause() -- we have to pause to manually step the animator

local Players = game:GetService("Players")
local function SpawnR15Character(spawnCFrame: CFrame)
    -- Make a description (can be empty = default R15 body)
    local desc = Instance.new("HumanoidDescription")
    local character = Players:CreateHumanoidModelFromDescription(
        desc,
        Enum.HumanoidRigType.R15
    )

    character.Name = "CustomR15"
    character.Parent = workspace
    if character.PrimaryPart then
        character:SetPrimaryPartCFrame(spawnCFrame)
    else
        -- Fallback: Move via HumanoidRootPart if PrimaryPart isn't set yet
        local hrp = character:FindFirstChild("HumanoidRootPart")
        if hrp then
            hrp.CFrame = spawnCFrame
        end
    end

    return character
end

local character = SpawnR15Character(CFrame.new(0, 0, 0))
local humanoid = character.Humanoid
local animator = humanoid.Animator
local animation = Instance.new("Animation")
local animationTrack = nil
print(character)

for fileData in FileSystemService:Walk("C:\\git\\roblox\\jrein\\anim-simularity\\out\\clips\\", Enum.FileSystemWalkMode.NonRecursive) do
	print(fileData.Path)
	
	local clip = FileSystemService:LoadInstances(fileData.Path)[1]

	local localClipId = KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
	local animation = Instance.new("Animation")
	animation.AnimationId = localClipId
	
	local animationTrack = animator:LoadAnimation(animation)
	animationTrack:Play(0)
	animationTrack.Looped = true -- must loop or stop with a fade time will get called and mess things up
	wait(0) -- we have to let the DM jobs step for the clip to load

	-- step through each frame
	local duration = animationTrack.Length
	local numSteps = math.floor(duration * 30 + 0.5)
	for i = 0, numSteps, 1.0 do
    	animationTrack.TimePosition = i / 30.0
		animator:StepAnimations(0)
		print(i, character.LeftHand.CFrame)
	end

	-- clean up
	animationTrack:Stop(0)
	animationTrack:Destroy()
	animationTrack = nil
	clip:Destroy()
	
	local tracks = animator:GetPlayingAnimationTracks()
	for _, track in pairs(tracks) do
		track:Stop(0)
		track:Destroy()	
	end
	animator:StepAnimations(0) -- allow cleanup to happen
	
	-- make sure no tracks are leaking!
	if #animator:GetPlayingAnimationTracks() > 0 then
		print("BAD TRACK NUM")
		local tracks = animator:GetPlayingAnimationTracks()
		for _, track in pairs(tracks) do
			print(track.Name, track.TimePosition)
		end
	end
	
	break
end

--FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\fingerprints4.csv", csv, Enum.FileMode.Text)