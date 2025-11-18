--!strict
-- Module: AnimSim
-- Similarity & dedup for CurveAnimation clips.

export type CanonFrame = {
	pos: Vector3,     -- local-space
	rot: CFrame,      -- local-space orientation (we'll store quaternion)
	scl: Vector3,     -- local-space scale if available (else Vector3.new(1,1,1))
}

export type TrackSeries = { CanonFrame }               -- [frame]
export type ClipSeries = { [string]: TrackSeries }     -- boneName -> TrackSeries

export type CanonParams = {
	FPS: number?,                 -- default 60
	DurationMode: "NormalizeTo1" | "Preserve", -- default "NormalizeTo1"
	QuantPos: number?,            -- meters (default 1e-3)
	QuantQuat: number?,           -- quaternion bin (default 1e-4)
	IncludeScale: boolean?,       -- default false
	--Mirror: boolean?,             -- produce mirrored variant (off by default)
	TrackOrder: {string}?,        -- canonical track order; if nil, inferred sorted keys
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

-- Quaternion helpers 
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
-- Adapter: sample CurveAnimation
-- ==============================
-- You must implement this for your CurveAnimation authoring setup.
-- Given (curveAnim, boneName, tSeconds), return local-space transform for that bone at t:
local function SampleTrackTransform(track: {}, t: number): CanonFrame
	
	if not track then
		return {
			pos = Vector3.zero,
			rot = CFrame.new(),
			scl = Vector3.new(1,1,1),
		}
	end

	return {
		pos = getCurveValueAtTime(track.pos, t),
		rot = getCurveValueAtTime(track.rot, t),
		scl = Vector3.new(1,1,1),
	}
end

-- ==============================
-- Canonicalization & resampling
-- ==============================
local function buildTrackList(curveAnim: Instance, explicitOrder:{string}?): {string}
	-- supply an explicit canonical list.
	if explicitOrder and #explicitOrder > 0 then
		return table.clone(explicitOrder)
	end

	-- return a common humanoid subset:
	local R15BoneNames = {
		"HumanoidRootPart","LowerTorso","UpperTorso","Head",
		"LeftUpperArm","LeftLowerArm","LeftHand",
		"RightUpperArm","RightLowerArm","RightHand",
		"LeftUpperLeg","LeftLowerLeg","LeftFoot",
		"RightUpperLeg","RightLowerLeg","RightFoot",
	}
	return R15BoneNames
end

local function canonicalize(curveAnim: Instance, duration:number, params: CanonParams?): (ClipSeries, number, {string})
	params = params or {}
	local FPS = params.FPS or 60
	local quantPos = params.QuantPos or 1e-3
	local quantQuat = params.QuantQuat or 1e-4
	local durationMode = params.DurationMode or "NormalizeTo1"
	local includeScale = params.IncludeScale or false
	local framesOverride = params.FramesOverride

	local tracks = buildTrackList(curveAnim, params.TrackOrder)
	local out: ClipSeries = {}
	
	local tracksCurves = getCurveTracks(curveAnim)

	-- Normalize duration & frame count
	local normDuration = (durationMode == "NormalizeTo1") and 1.0 or duration
	local numFrames = framesOverride or math.max(1, math.floor(normDuration * FPS + 0.5))
	local dt = normDuration / math.max(1, numFrames-1)

	for _, bone in ipairs(tracks) do
		local series: TrackSeries = table.create(numFrames)
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
			if includeScale then
				sx = quantizeFloat(fr.scl.X, 1e-3)
				sy = quantizeFloat(fr.scl.Y, 1e-3)
				sz = quantizeFloat(fr.scl.Z, 1e-3)
			end

			series[i+1] = {
				pos = Vector3.new(px,py,pz),
				rot = quatToCFrame(qx,qy,qz,qw), 
				scl = Vector3.new(sx,sy,sz),
			}
		end
		out[bone] = series
	end

	return out, numFrames, tracks
end

-- Serialize canonical clip to bytes in stable order -------------
local function serializeClip(cs: ClipSeries, tracks:{string}, includeScale:boolean): string
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
			if includeScale then
				table.insert(chunks, writeUInt32LE(fr.scl.X))
				table.insert(chunks, writeUInt32LE(fr.scl.Y))
				table.insert(chunks, writeUInt32LE(fr.scl.Z))
			end
		end
	end
	return table.concat(chunks)
end

-- Per-track exact hash + SimHash over shingles ------------------
local function perTrackHashes(cs: ClipSeries, tracks:{string})
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

-- 128D embedding with pose + motion
local function buildEmbedding(cs: ClipSeries, tracks:{string})
	-- Collect features in a deterministic bone order
	local feat = {}

	local function push(x:number) table.insert(feat, x) end
	local function push3(x:number,y:number,z:number) push(x); push(y); push(z) end

	-- Choose quartile sample frame indices
	local function quartileFrames(n:number): {number}
		if n <= 1 then return {1,1,1,1,1} end
		local last = n
		return {
			1,
			math.max(1, math.floor(0.25*(last-1)+1 + 0.5)),
			math.max(1, math.floor(0.50*(last-1)+1 + 0.5)),
			math.max(1, math.floor(0.75*(last-1)+1 + 0.5)),
			last,
		}
	end

	-- Pass 1: precompute quaternion arrays per bone and (optionally) relative-to-parent quats
	local quats: {[string]: {{x:number,y:number,z:number,w:number}}} = {}
	for _,bone in ipairs(tracks) do
		local s = cs[bone] or {}
		local qarr = table.create(#s)
		for i=1,#s do
			local qx,qy,qz,qw = cframeToQuat(s[i].rot)
			-- Align hemisphere over time against the first frame for stability
			if i == 1 then
				qarr[i] = {x=qx,y=qy,z=qz,w=qw}
			else
				local ax,ay,az,aw = qarr[1].x, qarr[1].y, qarr[1].z, qarr[1].w
				local bx,by,bz,bw = quatHemisphereAlign(ax,ay,az,aw, qx,qy,qz,qw)
				qarr[i] = {x=bx,y=by,z=bz,w=bw}
			end
		end
		quats[bone] = qarr
	end

	-- Optional: parent-relative quats (if your cs[bone].rot is not already local to parent)
	local relQuat: {[string]: {{x:number,y:number,z:number,w:number}}} = {}
	do
		for _,bone in ipairs(tracks) do
			local parent = PARENT and PARENT[bone] or nil
			local s = cs[bone] or {}
			local r = table.create(#s)
			if parent and cs[parent] and #cs[parent] == #s then
				for i=1,#s do
					-- q_rel = inv(q_parent) * q_child
					local px,py,pz,pw = cframeToQuat(cs[parent][i].rot)
					local cx,cy,cz,cw = quats[bone][i].x,quats[bone][i].y,quats[bone][i].z,quats[bone][i].w
					-- inverse parent: ( -p.xyz, p.w )
					local ix,iy,iz,iw = -px,-py,-pz,pw
					-- Hamilton product
					local rx = iw*cx + ix*cw + iy*cz - iz*cy
					local ry = iw*cy - ix*cz + iy*cw + iz*cx
					local rz = iw*cz + ix*cy - iy*cx + iz*cw
					local rw = iw*cw - ix*cx - iy*cy - iz*cz
					-- Hemisphere align to first
					if i>1 then
						local ax,ay,az,aw = r[1].x,r[1].y,r[1].z,r[1].w
						rx,ry,rz,rw = quatHemisphereAlign(ax,ay,az,aw, rx,ry,rz,rw)
					end
					r[i] = {x=rx,y=ry,z=rz,w=rw}
				end
			else
				-- Fallback: use absolute (already hemisphere-aligned) quats
				for i=1,#s do
					local q = quats[bone][i]
					-- Align to first for stability
					if i>1 then
						local ax,ay,az,aw = quats[bone][1].x,quats[bone][1].y,quats[bone][1].z,quats[bone][1].w
						local bx,by,bz,bw = quatHemisphereAlign(ax,ay,az,aw, q.x,q.y,q.z,q.w)
						r[i] = {x=bx,y=by,z=bz,w=bw}
					else
						r[i] = {x=q.x,y=q.y,z=q.z,w=q.w}
					end
				end
			end
			relQuat[bone] = r
		end
	end

	-- Pass 2: build features per bone
	for _,bone in ipairs(tracks) do
		local s = cs[bone] or {}
		local n = #s

		-- --- Pose: mean quaternion log (relative to parent if available)
		do
			local meanX, meanY, meanZ = 0,0,0
			for i=1,n do
				local q = relQuat[bone][i]
				local lx,ly,lz = quatLog(q.x,q.y,q.z,q.w)
				meanX += lx; meanY += ly; meanZ += lz
			end
			if n > 0 then meanX /= n; meanY /= n; meanZ /= n end
			push3(meanX, meanY, meanZ)
		end

		-- --- Pose snapshots at quartiles: quat-log + local pos
		do
			local idxs = quartileFrames(n)
			for _,ii in ipairs(idxs) do
				local q = relQuat[bone][ii]
				local lx,ly,lz = quatLog(q.x,q.y,q.z,q.w)
				push3(lx,ly,lz)
				local p = s[ii].pos
				-- Center positions to tame offsets (subtract mean pos over clip)
				-- Compute mean pos quickly (single pass)
				-- (We can approximate by using frame 1 as center if n is small)
				push3(p.X, p.Y, p.Z)
			end
		end

		-- --- Motion: velocity energies (same idea as before, but compact)
		do
			local bins = 4
			local eX = table.create(bins, 0.0)
			local eY = table.create(bins, 0.0)
			local eZ = table.create(bins, 0.0)
			local prev = (n>0) and s[1].pos or Vector3.zero
			for i=2,n do
				local v = s[i].pos - prev
				prev = s[i].pos
				local b = ((i-2) * bins) // math.max(1,(n-2))
				if b < 0 then b = 0 elseif b >= bins then b = bins-1 end
				eX[b+1] += math.abs(v.X)
				eY[b+1] += math.abs(v.Y)
				eZ[b+1] += math.abs(v.Z)
			end
			for b=1,bins do push(eX[b]); push(eY[b]); push(eZ[b]) end
		end

		-- --- Stillness scalar (helps low-motion clips cluster by pose)
		do
			local var = 0.0
			for i=2,n do
				local dp = s[i].pos - s[i-1].pos
				var += dp:Dot(dp)
			end
			local still = 1.0 / (1.0 + var) -- ≈1.0 if almost static, →0 with motion
			push(still)
		end
	end

	-- ===== Post-process: size control to 128D and L2 normalize ====
	-- If too few tracks/frames, we’ll have <128; pad with zeros.
	while #feat < 128 do table.insert(feat, 0.0) end
	if #feat > 128 then
		for i=129,#feat do
			local j = ((i-1) % 128) + 1
			feat[j] += feat[i]
		end
		for i=#feat,129,-1 do table.remove(feat) end
	end

	-- Reweight pose heavier than motion in low-movement clips:
	-- We infer a global stillness ~ mean of per-bone stillness scalars,
	-- then softly amplify the first pose-heavy block.
	do
		local sumStill, cnt = 0.0, 0
		-- Each bone contributed: [mean quat-log (3)] + [4* (quat-log(3)+pos(3)) = 24] + [motion 12] + [still 1] = 40 dims per bone
		-- The last of each 40 is 'still'.
		local stride = 40
		for bi=1,#tracks do
			local stillIdx = bi*stride
			if feat[stillIdx] ~= nil then
				sumStill += feat[stillIdx]; cnt += 1
			end
		end
		local globalStill = (cnt>0) and (sumStill/cnt) or 0.0 -- in [0,1]
		-- Scale the first 27 dims per bone (pose parts) by (1 + 0.5*globalStill)
		local poseBoost = 1.0 + 0.5*globalStill
		for bi=0,#tracks-1 do
			for k=1,27 do
				local idx = bi*stride + k
				if feat[idx] ~= nil then feat[idx] *= poseBoost end
			end
		end
	end

	-- L2 normalize
	local acc = 0.0
	for _,x in ipairs(feat) do acc += x*x end
	local inv = (acc>0) and (1.0/math.sqrt(acc)) or 1.0
	for i=1,#feat do feat[i] *= inv end

	return feat
end


-- Public: fingerprint a CurveAnimation --------------------------
function AnimSim.Fingerprint(curveAnim: Instance, duration:number, params: CanonParams?): Fingerprint
	local includeScale = (params and params.IncludeScale) or false
	local cs, numFrames, tracks = canonicalize(curveAnim, duration, params)
	local bytes = serializeClip(cs, tracks, includeScale)
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

-- Precise curve similarity --------------------------------------
local function perTrackCurveSimilarity(sa:TrackSeries, sb:TrackSeries, allowPhase:boolean, maxPhase:number)
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

function AnimSim.Score(curveA: Instance, durA:number, curveB: Instance, durB:number, trackOrder:{string}?, scoreParams: ScoreParams?): SimilarityBreakdown
	scoreParams = scoreParams or {}
	local allowPhase = (scoreParams.AllowPhaseShift ~= false)
	local maxPhase = scoreParams.MaxPhaseFrames or 10
	local wRot = scoreParams.RotationWeight or 0.6
	local wPos = scoreParams.PositionWeight or 0.4
	local effW = scoreParams.EndEffectorWeights or {}

	local canonParams: CanonParams = {
		FPS = 60, DurationMode = "NormalizeTo1",
		QuantPos = 1e-3, QuantQuat = 1e-4,
		IncludeScale = false, TrackOrder = trackOrder,
	}
	local csA, _, tracks = canonicalize(curveA, durA, canonParams)
	local csB, _, _ = canonicalize(curveB, durB, canonParams)

	local perTrack = {}
	local bestPhaseGlobal = 0
	local weightedSum = 0.0
	local weightTotal = 0.0

	local worst = {}
	for _,bone in ipairs(tracks) do
		local sA = csA[bone]; local sB = csB[bone]
		if sA and sB then
			local best, sPos, sRot = perTrackCurveSimilarity(sA, sB, allowPhase, maxPhase)
			local w = (effW[bone] or 1.0)
			local s = w * (wRot*sRot + wPos*sPos)
			perTrack[bone] = s
			weightedSum += s
			weightTotal += w
			table.insert(worst, {bone=bone, val=s})
		else
			perTrack[bone] = 0
		end
	end
	table.sort(worst, function(a,b) return a.val < b.val end)
	local worstList = {}
	for i=1, math.min(5, #worst) do table.insert(worstList, worst[i].bone) end

	local score = (weightTotal>0) and (weightedSum / weightTotal) or 0.0
	return {
		score = score,
		perTrack = perTrack,
		bestPhase = bestPhaseGlobal, -- per-track phases kept internal; can be exposed if needed
		worstTracks = worstList,
	}
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


local csv = "animId,clipId,duration,hash,embedding1-128,\n"

local count = 0
local FileSystemService = game:GetService("FileSystemService")
for fileData in FileSystemService:Walk("C:\\git\\roblox\\anim-similarity\\out\\clips\\", Enum.FileSystemWalkMode.NonRecursive) do
	--print(fileData.Path)
	local clip = FileSystemService:LoadInstances(fileData.Path)[1]
	if not clip or not clip:IsA("CurveAnimation") then
		print("clip is not a curve animation", clip, clip.ClassName, fileData.Path)
		continue
	end
	
	local animId, clipId = fileData.Path:match(".*/(%d+)%-(%d+)%.rbxm$")

	local success, result = pcall(function()
		local duration = calculateCurveAnimLength(clip)
		local fingerprint = AnimSim.Fingerprint(clip, duration, { FPS = 120, DurationMode = "NormalizeTo1" })
		local line = animId .. "," .. clipId .. "," .. duration .. "," .. fingerprint.clipHash .. ","
		for i, n in fingerprint.embedding do
			line = line .. n .. ","
		end
		line = line .. "\n"
		
		--print(line)
		csv = csv .. line
	end)
	
	count = count + 1
	
	if count % 100 == 0 then
		print(count)
	end
	
	clip:Destroy()
end

FileSystemService:WriteFile("C:\\git\\roblox\\anim-similarity\\fingerprints3.csv", csv, Enum.FileMode.Text)