require 'nn'
local utils = require 'utils'
local classifier = {}

function classifier.setOpts(opt)
   classifier.sigmoid_wt =  opt.sigmoid_wt 
   classifier.seq_wt = opt.seq_wt 
   classifier.vocab_size = opt.classifier_vocab_size
   classifier.xdim = opt.feature_xdim
   classifier.loader_info = opt.loader_info
end

function classifier.init(cnn,rnn,mlp)
  classifier.cnn = cnn
  classifier.rnn = rnn
  classifier.mlp = mlp
  classifier.sigmoid = nn.Sigmoid():type(mlp:type())
end

function get_seq_prob(seq_l)
   seq_prob = torch.zeros(classifier.vocab_size)
   for i=1,classifier.vocab_size do
     local pw = 1./3.
     local u = classifier.loader_info.unigrams[i]/classifier.loader_info.num_instances
     local b = 0
     if classifier.loader_info.unigrams[seq_l[1]] ~= 0 then 
        b = classifier.loader_info.bigrams[i][seq_l[1]]/classifier.loader_info.unigrams[seq_l[1]]
     end
     local t = 0
     if classifier.loader_info.bigrams[seq_l[1]][seq_l[2]] ~= 0 then
        t = classifier.loader_info.trigrams[i][seq_l[1]][seq_l[2]]/classifier.loader_info.bigrams[seq_l[1]][seq_l[2]]
     end
             
     seq_prob[i] = pw*u + pw*b + pw*t
                       
   end
   return seq_prob
end

function smooth_with_seq(sigmoid_out)
   local Y, cls_label= torch.sort(sigmoid_out,1,true)
   cls_label = cls_label[{{1,2}}]
   local seq_prob = get_seq_prob(cls_label)

   sigmoid_out = sigmoid_out:view(-1)
   seq_prob = seq_prob:type(sigmoid_out:type())
   local smooth_out = sigmoid_out:mul(classifier.sigmoid_wt) + seq_prob:mul(classifier.seq_wt)
   return smooth_out
end

function classifier.forward(input,add)

   local output = {}
   local num_clips = input:size(1)
   local feat_len = input:size(3)
   
   input = utils.adjustToSize(input,classifier.xdim)
   local norm = classifier.cnn:get(1):forward(input:view(num_clips*feat_len,classifier.xdim))
   norm = norm:view(num_clips,1,feat_len,classifier.xdim)
   local cnn_output = classifier.cnn:get(2):forward(norm)
   cnn_output = cnn_output:view(cnn_output:size(1),-1)

   local rnn_in = utils.tensor_to_table(cnn_output)
   local rnn_output = classifier.rnn:forward(rnn_in)

   local linear_output = classifier.mlp:forward(rnn_output)
   local sigmoid_out = classifier.sigmoid:forward(linear_output)
   local smooth_out = smooth_with_seq(sigmoid_out)

   --sort the results and choose the top  10 results greater than certain thresh
   local Y, cls_label= torch.sort(smooth_out,1,true)
   if cls_label:numel() > 10 then cls_label = cls_label[{{1,10}}] end
   smooth_out = smooth_out:index(1,cls_label)   
   for i = 1,smooth_out:size(1) do
     if smooth_out[i] > 0.1 then output[cls_label[i]] = smooth_out[i] end
   end

   return output
end

function classifier.clearState()
   classifier.cnn:clearState()
   classifier.rnn:clearState()
   classifier.mlp:clearState()
   classifier.sigmoid:clearState()
end

return classifier
