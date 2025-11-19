--!strict
-- Module: AnimSim
-- Similarity & dedup for CurveAnimation clips.


--"run --run C:\\git\\roblox\\jrein\\anim-simularity\\fingerprinting.lua --fs.readwrite C:\\git\\roblox\\jrein\\anim-simularity\\ --load.asRobloxScript"

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
	--Mirror: boolean?,             -- produce mirrored variant (off by default)
	TrackOrder: {string}?,        -- canonical track order; if nil R15
	FramesOverride: number?,      -- optional fixed frame count after normalization
}

export type Fingerprint = {
	clipHash: number,                 -- exact hash of canonical bytes (FNV1a-32)
	perTrackExact: { [string]: number },
	perTrackPhash:  { [string]: number },
	embedding: { number },            -- 128D
	meta: {
		numFrames: number,
		fps: number,
		tracks: {string},
		duration: number,
	},
}

export type ScoreParams = {
	RotationWeight: number?,  -- default 0.6
	PositionWeight: number?,  -- default 0.4
	EndEffectorWeights: { [string]: number }?, -- e.g., Hands/Feet 2.0
	AllowPhaseShift: boolean?, -- default true
	MaxPhaseFrames: number?,   -- default 10
}

export type SimilarityBreakdown = {
	score: number,                -- 0..1
	perTrack: { [string]: number },
	bestPhase: number,
	worstTracks: {string},
}

local AnimSim = {}

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

local function quatDot(ax,ay,az,aw, bx,by,bz,bw)
	return ax*bx + ay*by + az*bz + aw*bw
end

local function quatSlerp(ax,ay,az,aw, bx,by,bz,bw, t:number)
	-- Handles antipodal
	local cosom = quatDot(ax,ay,az,aw, bx,by,bz,bw)
	if cosom < 0 then
		bx,by,bz,bw = -bx,-by,-bz,-bw
		cosom = -cosom
	end
	if cosom > 0.9995 then
		-- linear fallback
		local x = ax + t*(bx-ax)
		local y = ay + t*(by-ay)
		local z = az + t*(bz-az)
		local w = aw + t*(bw-aw)
		local inv = 1.0 / math.sqrt(x*x+y*y+z*z+w*w)
		return x*inv, y*inv, z*inv, w*inv
	end
	local omega = math.acos(clamp(cosom, -1, 1))
	local sinom = math.sin(omega)
	local s0 = math.sin((1.0 - t)*omega) / sinom
	local s1 = math.sin(t*omega) / sinom
	local x = s0*ax + s1*bx
	local y = s0*ay + s1*by
	local z = s0*az + s1*bz
	local w = s0*aw + s1*bw
	return x,y,z,w
end

local function quatAngle(ax,ay,az,aw, bx,by,bz,bw)
	local d = math.abs(quatDot(ax,ay,az,aw, bx,by,bz,bw))
	d = clamp(d, -1, 1)
	return math.acos(d) -- radians
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

-- Little-endian 32-bit writer
local function writeUInt32LE(n:number): string
	n = u32(n)
	local b0 = n % 256
	local b1 = math.floor(n / 256) % 256
	local b2 = math.floor(n / 65536) % 256
	local b3 = math.floor(n / 16777216) % 256
	return string.char(b0, b1, b2, b3)
end

-- FNV-1a 32 using bxor + mul with wrap
local function fnv1a32(bytes:string): number
	local hash = 0x811C9DC5
	for i = 1, #bytes do
		hash = bit32.bxor(hash, string.byte(bytes, i))
		hash = u32(hash * 16777619)
	end
	return hash
end

local function simhash32(features:{number}): number
	local accum = table.create(32, 0.0)

	local function proj(i:number, bit:number)
		-- deterministic pseudo-random sign via LCG + bit32
		local v = bit32.rshift((1103515245 * (i + bit*97) + 12345), 16)
		v = bit32.band(v, 0x7FFF)
		return (bit32.band(v, 1) == 0) and 1.0 or -1.0
	end

	for idx, f in ipairs(features) do
		for b = 0, 31 do
			accum[b+1] += proj(idx, b) * f
		end
	end

	local h = 0
	for b = 0, 31 do
		if accum[b+1] >= 0 then
			h = bit32.bor(h, bit32.lshift(1, b))
		end
	end
	return h
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

local function quatHemisphereAlign(ax,ay,az,aw, bx,by,bz,bw)
	-- Flip b if dot(a,b) < 0 so logs average consistently
	local dot = ax*bx + ay*by + az*bz + aw*bw
	if dot < 0 then
		return -bx, -by, -bz, -bw
	end
	return bx, by, bz, bw
end

local function quatLog(x,y,z,w) -- maps unit quat -> R^3 (axis * angle)
	-- Ensure normalized-ish
	local v2 = x*x + y*y + z*z
	if v2 < 1e-12 then
		return 0,0,0
	end
	local v = math.sqrt(v2)
	-- angle = 2 * atan2(|v|, w)
	local ang = 2.0 * math.atan2(v, w)
	local s = ang / v
	return x*s, y*s, z*s
end

-- ==============================
-- Canonicalization & resampling
-- ==============================
local function buildTrackList(explicitOrder:{string}?): {string}
	-- supply an explicit canonical list.
	if explicitOrder and #explicitOrder > 0 then
		return table.clone(explicitOrder)
	end

	-- return R15:
	local R15BoneNames = {
		"LowerTorso","UpperTorso","Head",
		"LeftUpperArm","LeftLowerArm","LeftHand",
		"RightUpperArm","RightLowerArm","RightHand",
		"LeftUpperLeg","LeftLowerLeg","LeftFoot",
		"RightUpperLeg","RightLowerLeg","RightFoot",
	}
	return R15BoneNames
end

local function SampleTrackTransform(bone: string): AnimTransform

	return {
		pos = character[bone].CFrame.Position,
		rot = character[bone].CFrame.Rotation,
	}
end

local function canonicalize(duration:number, params: FingerprintParams?): (ClipData, number, {string})
	
	if #animator:GetPlayingAnimationTracks() ~= 1 then
		print("ERROR: THERE SHOULD BE ONE TRACK")
	end
	
	params = params or {}
	local FPS = params.FPS or 60
	local quantPos = params.QuantPos or 1e-3
	local quantQuat = params.QuantQuat or 1e-4
	local durationMode = params.DurationMode or "NormalizeTo1"
	local framesOverride = params.FramesOverride

	local tracks = buildTrackList(params.TrackOrder)
	local out: ClipData = {}
	
	-- Normalize duration & frame count
	local normDuration = (durationMode == "NormalizeTo1") and 1.0 or duration
	local numFrames = framesOverride or math.max(1, math.floor(normDuration * FPS + 0.5))
	local dt = normDuration / math.max(1, numFrames-1)
	
	-- setup out data
	for _, bone in ipairs(tracks) do
		local frames: TrackFrames = table.create(numFrames)
		out[bone] = frames
	end
	
	for i=0,numFrames-1 do
		local t = i * dt
		-- Map resampled time back to source duration if preserving:
		local srcT = (durationMode == "NormalizeTo1") and (t * duration) or t
		
		-- update our track and pose
		animationTrack.TimePosition = srcT
		animator:StepAnimations(0)
		
		for idx, bone in ipairs(tracks) do
			local fr = SampleTrackTransform(bone)

			-- Quantize for stability
			local qx,qy,qz,qw = cframeToQuat(fr.rot)			
			qx = quantizeFloat(qx, params.QuantQuat or quantQuat)
			qy = quantizeFloat(qy, params.QuantQuat or quantQuat)
			qz = quantizeFloat(qz, params.QuantQuat or quantQuat)
			qw = quantizeFloat(qw, params.QuantQuat or quantQuat)
			local px = quantizeFloat(fr.pos.X, quantPos)
			local py = quantizeFloat(fr.pos.Y, quantPos)
			local pz = quantizeFloat(fr.pos.Z, quantPos)

			--if idx == 6 then
			--	print(bone, i, px,py,pz, qx,qy,qz,qw)
			--end

			out[bone][i+1] = {
				pos = Vector3.new(px,py,pz),
				rot = quatToCFrame(qx,qy,qz,qw), 
			}

		end	
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

-- Per-track exact hash + SimHash over shingles ------------------
local function perTrackHashes(cs: ClipData, tracks:{string})
	local exact: { [string]: number } = {}
	local ph:    { [string]: number } = {}

	for _,bone in ipairs(tracks) do
		local s = cs[bone]
		-- exact: pack ints for pos+quat
		local parts = table.create(#s * 10)
		for _,fr in ipairs(s) do
			local qx,qy,qz,qw = cframeToQuat(fr.rot)
			table.insert(parts, writeUInt32LE(fr.pos.X)); table.insert(parts, writeUInt32LE(fr.pos.Y)); table.insert(parts, writeUInt32LE(fr.pos.Z))
			table.insert(parts, writeUInt32LE(quantizeFloat(qx, 1e-4)))
			table.insert(parts, writeUInt32LE(quantizeFloat(qy, 1e-4)))
			table.insert(parts, writeUInt32LE(quantizeFloat(qz, 1e-4)))
			table.insert(parts, writeUInt32LE(quantizeFloat(qw, 1e-4)))
		end
		exact[bone] = fnv1a32(table.concat(parts))

		-- phash: simple shingles of 8 frames using velocity deltas
		local feats = {}
		local W = 8
		for i=2,#s do
			local p0 = s[i-1].pos; local p1 = s[i].pos
			local dp = p1 - p0
			table.insert(feats, quantizeFloat(dp.X*1000, 1)) -- mm/s (relative to frame step)
			table.insert(feats, quantizeFloat(dp.Y*1000, 1))
			table.insert(feats, quantizeFloat(dp.Z*1000, 1))
			if (i % W) == 0 then
				-- flush a window boundary by adding a separator feature
				table.insert(feats, 123457 + i)
			end
		end
		ph[bone] = simhash32(feats)
	end
	return exact, ph
end

-- 128D embedding using object-space pose + motion
local function buildEmbedding(cs: ClipSeries, tracks:{string})
	local feat = {}
	local function push(x:number) table.insert(feat, x) end
	local function push3(x:number,y:number,z:number)
		table.insert(feat, x); table.insert(feat, y); table.insert(feat, z)
	end

	for _,bone in ipairs(tracks) do
		local s: TrackSeries = cs[bone]
		local n = s and #s or 0

		if n == 0 then
			-- No data for this bone: reserve a consistent slot
			-- mean pos(3) + mean rotLog(3) + var pos(3) + var rotLog(3)
			-- + vel energy(3) + stillness(1) = 16 dims
			for _=1,16 do push(0.0) end
		else
			-- First pass: get quats hemisphere-aligned vs first frame
			local qx0,qy0,qz0,qw0 = cframeToQuat(s[1].rot)
			local qlogs = table.create(n)
			local sumPX,sumPY,sumPZ = 0,0,0
			local sumLX,sumLY,sumLZ = 0,0,0

			for i=1,n do
				local fr = s[i]
				local px,py,pz = fr.pos.X, fr.pos.Y, fr.pos.Z

				-- object-space position (already, thanks to your change)
				sumPX += px; sumPY += py; sumPZ += pz

				local qx,qy,qz,qw = cframeToQuat(fr.rot)
				-- hemisphere-align to first quaternion for stability
				qx,qy,qz,qw = quatHemisphereAlign(qx0,qy0,qz0,qw0, qx,qy,qz,qw)
				local lx,ly,lz = quatLog(qx,qy,qz,qw)
				qlogs[i] = {lx=lx, ly=ly, lz=lz}
				sumLX += lx; sumLY += ly; sumLZ += lz
			end

			local invN = 1.0 / n
			local meanPX,meanPY,meanPZ = sumPX*invN, sumPY*invN, sumPZ*invN
			local meanLX,meanLY,meanLZ = sumLX*invN, sumLY*invN, sumLZ*invN

			-- Second pass: variance of pos + rotLog
			local varPX,varPY,varPZ = 0,0,0
			local varLX,varLY,varLZ = 0,0,0
			for i=1,n do
				local fr = s[i]
				local dx = fr.pos.X - meanPX
				local dy = fr.pos.Y - meanPY
				local dz = fr.pos.Z - meanPZ
				varPX += dx*dx; varPY += dy*dy; varPZ += dz*dz

				local ql = qlogs[i]
				local dlx = ql.lx - meanLX
				local dly = ql.ly - meanLY
				local dlz = ql.lz - meanLZ
				varLX += dlx*dlx; varLY += dly*dly; varLZ += dlz*dlz
			end
			varPX *= invN; varPY *= invN; varPZ *= invN
			varLX *= invN; varLY *= invN; varLZ *= invN

			-- Motion: velocity energy in object-space pos
			local velEX,velEY,velEZ = 0,0,0
			local accVar = 0.0 -- for stillness
			if n > 1 then
				local prev = s[1].pos
				for i=2,n do
					local p = s[i].pos
					local vx = p.X - prev.X
					local vy = p.Y - prev.Y
					local vz = p.Z - prev.Z
					prev = p
					velEX += math.abs(vx)
					velEY += math.abs(vy)
					velEZ += math.abs(vz)
					accVar += vx*vx + vy*vy + vz*vz
				end
				local invNSteps = 1.0 / (n-1)
				velEX *= invNSteps; velEY *= invNSteps; velEZ *= invNSteps
			end

			-- Stillness scalar: ~1.0 if almost no movement, ->0 if lots of motion
			local still = 1.0 / (1.0 + accVar)

			-- Push features for this bone (16 dims):
			-- mean pos (3)
			push3(meanPX, meanPY, meanPZ)
			-- mean rotLog (3)
			push3(meanLX, meanLY, meanLZ)
			-- var pos (3)
			push3(varPX, varPY, varPZ)
			-- var rotLog (3)
			push3(varLX, varLY, varLZ)
			-- vel energy (3)
			push3(velEX, velEY, velEZ)
			-- stillness (1)
			push(still)
		end
	end

	-- At this point we have 16 dims per bone.
	-- For 15 bones, that’s 240 dims. Fold/pad to 128 then L2-normalize.

	-- Pad or fold down to exactly 128 dims
	while #feat < 128 do
		table.insert(feat, 0.0)
	end
	if #feat > 128 then
		for i = 129, #feat do
			local j = ((i-1) % 128) + 1
			feat[j] += feat[i]
		end
		for i = #feat, 129, -1 do
			table.remove(feat, i)
		end
	end

	-- L2 normalize for cosine similarity
	local acc = 0.0
	for _,x in ipairs(feat) do
		acc += x*x
	end
	local inv = (acc > 0) and (1.0 / math.sqrt(acc)) or 1.0
	for i=1,#feat do
		feat[i] *= inv
	end

	return feat
end

-- Public: fingerprint --------------------------
function AnimSim.Fingerprint(duration:number, params: FingerprintParams?): Fingerprint
	local cs, numFrames, tracks = canonicalize(duration, params)
	local bytes = serializeClip(cs, tracks)
	local clipHash = fnv1a32(bytes)
	--local perExact, perPhash = perTrackHashes(cs, tracks)
	local embed = buildEmbedding(cs, tracks)

	return {
		clipHash = clipHash,
		--perTrackExact = perExact,
		--perTrackPhash = perPhash,
		embedding = embed,
		meta = {
			numFrames = numFrames,
			fps = (params and params.FPS) or 60,
			tracks = tracks,
			duration = (params and (params.DurationMode == "NormalizeTo1")) and 1.0 or duration,
		}
	}
end

-- Cosine similarity of embeddings -------------------------------
local function cosine(a:{number}, b:{number})
	local s, na, nb = 0, 0, 0
	local n = math.min(#a,#b)
	for i=1,n do
		local x, y = a[i], b[i]
		s += x*y; na += x*x; nb += y*y
	end
	if na == 0 or nb == 0 then return 0 end
	return s / math.sqrt(na*nb)
end

-- Precise similarity --------------------------------------
local function perTrackCurveSimilarity(sa:TrackFrames, sb:TrackFrames, allowPhase:boolean, maxPhase:number)
	-- We compute RMSE for position and quaternion angle, optionally over best phase shift.
	local function scoreForPhase(phase:number)
		local cnt = 0
		local sumPos = 0.0
		local sumRot = 0.0
		for i=1,#sa do
			local j = i + phase
			if j >=1 and j <= #sb then
				local a = sa[i]; local b = sb[j]
				local dp = a.pos - b.pos
				sumPos += dp:Dot(dp)
				local ax,ay,az,aw = cframeToQuat(a.rot)
				local bx,by,bz,bw = cframeToQuat(b.rot)
				sumRot += quatAngle(ax,ay,az,aw, bx,by,bz,bw)
				cnt += 1
			end
		end
		if cnt == 0 then return 0 end
		local rmsePos = math.sqrt(sumPos / cnt)        -- studs
		local meanRot = sumRot / cnt                    -- radians
		-- Convert to [0,1] similarities with soft decay
		local sPos = math.exp(- (rmsePos * 4.0))       -- tune 4.0 as scale
		local sRot = math.exp(- (meanRot * 2.0))       -- tune 2.0 as scale
		return sPos, sRot
	end

	local best = -1
	local bestPos, bestRot = 0.0, 0.0
	if allowPhase then
		for ph=-maxPhase, maxPhase do
			local sp, sr = scoreForPhase(ph)
			local comb = 0.6*sr + 0.4*sp
			if comb > best then best = comb; bestPos, bestRot = sp, sr end
		end
	else
		bestPos, bestRot = scoreForPhase(0)
		best = 0.6*bestRot + 0.4*bestPos
	end
	return best, bestPos, bestRot
end

-- Cheap candidate check: exact or near-dup ----------------------
function AnimSim.AreIdentical(fpa: Fingerprint, fpb: Fingerprint): boolean
	if fpa.clipHash == fpb.clipHash then return true end
	local same, total = 0, 0
	for _,name in ipairs(fpa.meta.tracks) do
		if fpb.perTrackExact[name] ~= nil then
			total += 1
			if fpb.perTrackExact[name] == fpa.perTrackExact[name] then same += 1 end
		end
	end
	return (total > 0) and (total - same <= 2)
end

function AnimSim.EmbeddingSimilarity(fpa: Fingerprint, fpb: Fingerprint): number
	return cosine(fpa.embedding, fpb.embedding)
end

print("Running:")

files = {{Path = "C:/git/roblox/jrein/anim-simularity/out/clips/104457080832465-75138815689213.rbxm"},
	     {Path = "C:/git/roblox/jrein/anim-simularity/out/clips/104458334783090-133164677339371.rbxm"}}

local csv = ""--"animId,clipId,duration,hash,embedding1-128,\n"
local count = 0
--for _, fileData in ipairs(files) do
for fileData in FileSystemService:Walk("C:\\git\\roblox\\jrein\\anim-simularity\\out\\clips\\", Enum.FileSystemWalkMode.NonRecursive) do
	--print(fileData.Path)
	local clip = FileSystemService:LoadInstances(fileData.Path)[1]
	
	local localClipId = KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
	animation.AnimationId = localClipId
	animationTrack = animator:LoadAnimation(animation)
	animationTrack.Name = fileData.Path
	animationTrack:Play(0)
	animationTrack.Looped = true -- must loop or stop with a fade time will get called and mess things up
	wait(0) -- we have to let the DM jobs step for the clip to load

	local duration = animationTrack.Length--calculateCurveAnimLength(clip)
	local animId, clipId = fileData.Path:match(".*/(%d+)%-(%d+)%.rbxm$")
	--print(animId, duration)

	local line = ""
	local success, result = pcall(function()
		local fingerprint = AnimSim.Fingerprint(duration, { FPS = 120, DurationMode = "NormalizeTo1" })
		line = animId .. "," .. clipId .. "," .. duration .. "," .. fingerprint.clipHash .. ","
		for i, n in fingerprint.embedding do
			line = line .. n .. ","
		end
		line = line .. "\n"
		csv = csv .. line
	end)
	count = count + 1
	
	if count % 100 == 0 then
		print(count)
		--print(line)
	end
	
	if count % 2000 == 0 then
		FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\fingerprints4.csv", csv, Enum.FileMode.Text)
	end
	
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
	
end

FileSystemService:WriteFile("C:\\git\\roblox\\jrein\\anim-simularity\\fingerprints4.csv", csv, Enum.FileMode.Text)