require 'torch'
require 'hdf5'
require 'nn'
require 'rnn'
local utils = require 'utils'

function parse(arg)
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Platform configs.')
  cmd:text()
  cmd:text('Options')
  -- Run time opts
  cmd:option('-i', '', 'input music file') 
  cmd:option('-m', 'models/seqloss_cnn_rnn_seq2one/wave01_f15.json.t7', 'input model') 
  cmd:option('-gpu', -1, 'which gpu to use. 0 / 1 = GPU ID')
  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

local opts = parse(arg)
if opts.gpu >= 0 then  
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
end

print("Loading model " .. opts.m)
local model = torch.load(opts.m)

local dtype = 'torch.FloatTensor'
torch.setdefaulttensortype(dtype)
torch.manualSeed(model.opt.seed)
if opts.gpu >= 0 then  
   cutorch.manualSeed(model.opt.seed)
   cutorch.setDevice(opts.gpu + 1) -- note +1 because lua is 1-indexed
   dtype = 'torch.CudaTensor'
end

local classifier = require(model.opt.classifier)
model.cnn:type(dtype)
model.rnn:type(dtype)
model.mlp:type(dtype)
model.cnn:evaluate()
model.rnn:evaluate()
model.mlp:evaluate()
classifier.setOpts(model.opt)
classifier.init(model.cnn, model.rnn, model.mlp)

print("Extracting spectogram ...")
os.execute('python audio_processor.py -i \"' .. opts.i .. "\"")
print("Done extracting")
local h5_cache = hdf5.open("cache.h5",'r')
local input = h5_cache:read("/1"):all()
local max_seq = model.opt.max_clips_per_song
input = input:type(dtype)
input = utils.splitInput(input,model.opt.feature_xdim,model.opt.feature_xdim,max_seq)

print("Solving ... ")
classifier.clearState()
local output = classifier.forward(input,nil)
print("Tags for " .. opts.i .. " : ")
for k,v in utils.spairs(output,function(t,a,b) return t[b] < t[a] end) do
  print(model.opt.loader_info.idx_to_token[k], v)
end

