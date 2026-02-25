--!strict
-- Load encoder weights (from encoder_weights.lua) and run MLP forward to get embedding.
-- Input: pose vector of length T_max * 90 (pose sequence, padded if needed). If encoder
-- was trained with energy, we append 15 per-bone energy values (accumulated delta pos + quatlog).
-- Output: embedding vector (latentDim).

local WEIGHTS_PATH = "ml_training/checkpoints/encoder_weights.lua"

-- Run a single linear layer: out = weight * in + bias. weight[out][in], 1-based.
local function linearForward(weight: { { number } }, bias: { number }, inputVec: { number }): { number }
	local outDim = #weight
	local inDim = #weight[1]
	local out: { number } = table.create(outDim)
	for o = 1, outDim do
		local sum = bias[o]
		for i = 1, inDim do
			sum = sum + weight[o][i] * inputVec[i]
		end
		out[o] = sum
	end
	return out
end

local function relu(x: number): number
	return math.max(0, x)
end

-- Forward through encoder: alternating Linear and ReLU, last layer Linear only.
local function encoderForward(weightsTable: any, inputVec: { number }): { number }
	local layers = weightsTable.layers
	local x = inputVec
	for L = 1, #layers do
		local layer = layers[L]
		x = linearForward(layer.weight, layer.bias, x)
		if L < #layers then
			for i = 1, #x do
				x[i] = relu(x[i])
			end
		end
	end
	return x
end

-- Load weights from the Lua file that returns a table (requires loadstring + ReadFile).
local function loadWeights(path: string): any
	local FileSystemService = game:GetService("FileSystemService")
	local content = FileSystemService:ReadFile(path, Enum.FileMode.Text)
	local fn = loadstring(content)
	if not fn then error("Failed to load weights file") end
	return fn()
end

local FEAT_PER_FRAME = 90
local NUM_BONES = 15
local ENERGY_DIM = 15

-- Per-bone energy: sum of frame-to-frame L2(delta pos) + L2(delta quatlog). poseVector length T_max*90.
local function computeBoneEnergy(poseVector: { number }, T_max: number): { number }
	local energy: { number } = table.create(ENERGY_DIM, 0)
	if T_max < 2 then
		return energy
	end
	for b = 0, NUM_BONES - 1 do
		local base = b * 6
		local sumPos, sumQuat = 0.0, 0.0
		for t = 1, T_max - 1 do
			local i0 = (t - 1) * FEAT_PER_FRAME + base
			local i1 = t * FEAT_PER_FRAME + base
			local dx = poseVector[i1 + 1] - poseVector[i0 + 1]
			local dy = poseVector[i1 + 2] - poseVector[i0 + 2]
			local dz = poseVector[i1 + 3] - poseVector[i0 + 3]
			sumPos = sumPos + math.sqrt(dx * dx + dy * dy + dz * dz)
			local lx = poseVector[i1 + 4] - poseVector[i0 + 4]
			local ly = poseVector[i1 + 5] - poseVector[i0 + 5]
			local lz = poseVector[i1 + 6] - poseVector[i0 + 6]
			sumQuat = sumQuat + math.sqrt(lx * lx + ly * ly + lz * lz)
		end
		energy[b + 1] = sumPos + sumQuat
	end
	return energy
end

-- Build embedding for a clip: poseVector is (T_max * 90) float array (pad with 0 if shorter).
-- If encoder was trained with energy (inputDim == T_max*90+15), we compute and append energy.
-- Returns embedding as array of length latentDim.
local function buildEmbeddingFromPoseVector(poseVector: { number }, weightsPath: string?): { number }
	local path = weightsPath or WEIGHTS_PATH
	local w = loadWeights(path)
	local T_max = w.T_max
	local frameLen = T_max * FEAT_PER_FRAME
	local inputDim = w.inputDim
	local inputVec: { number } = table.create(inputDim, 0)
	for i = 1, math.min(#poseVector, frameLen) do
		inputVec[i] = poseVector[i]
	end
	if inputDim > frameLen then
		local energy = computeBoneEnergy(inputVec, T_max)
		for i = 1, ENERGY_DIM do
			inputVec[frameLen + i] = energy[i]
		end
	end
	return encoderForward(w, inputVec)
end

-- Export for use by other scripts
return {
	loadWeights = loadWeights,
	encoderForward = encoderForward,
	buildEmbeddingFromPoseVector = buildEmbeddingFromPoseVector,
}
