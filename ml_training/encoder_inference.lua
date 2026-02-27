--!strict
-- Load encoder weights (from encoder_weights.lua) and run MLP forward to get embedding.
-- Input: pose vector of length T_max * 90 (pose sequence, padded if needed). Frames only; no energy.
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

-- Build embedding for a clip: poseVector is (T_max * 90) float array (pad with 0 if shorter).
-- Encoder input is frames only; no energy. Returns embedding as array of length latentDim.
local function buildEmbeddingFromPoseVector(poseVector: { number }, weightsPath: string?): { number }
	local path = weightsPath or WEIGHTS_PATH
	local w = loadWeights(path)
	local T_max = w.T_max
	local frameLen = T_max * FEAT_PER_FRAME
	local inputVec: { number } = table.create(frameLen, 0)
	for i = 1, math.min(#poseVector, frameLen) do
		inputVec[i] = poseVector[i]
	end
	return encoderForward(w, inputVec)
end

-- Export for use by other scripts
return {
	loadWeights = loadWeights,
	encoderForward = encoderForward,
	buildEmbeddingFromPoseVector = buildEmbeddingFromPoseVector,
}
