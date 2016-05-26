require 'rnn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'neuralconvo'
path = require 'pl.path'

local tokenizer = require "tokenizer"
local list = require "pl.List"

cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Data/model options**")
cmd:text("")

cmd:option('--model_file', "data/model.t7", [[Path to the model file]])
cmd:option('--input_file', "data/test_sentences.txt", [[Path to the input text file]])
cmd:option('--output_file', "data/embeddings.t7", [[Path to the output t7 file]])
cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
cmd:option('--opencl', false, 'use OpenCL. Training must be done on OpenCL')


function pred2sent(wordIds, i)
  local words = {}
  i = i or 1
  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId[i]]
    table.insert(words, word)
  end

  return tokenizer.join(words)
end


function encode(sentence)
  print(sentence)
  local wordIds = {}
  for t, word in tokenizer.tokenize(sentence) do
    local id = dataset.word2id[word:lower()] or dataset.unknownToken
    table.insert(wordIds, id)
  end
  local input = torch.Tensor(list.reverse(wordIds))
  return model.encoder:forward(input):clone():view(1, 300)
end

function get_tensor(input_table)
    m = nn.JoinTable(1)
    return m:forward(input_table)
end

function encode_file(file)
  local result = {}
  for line in file:lines()  do
    table.insert(result, encode(line))
  end
  return get_tensor(result)
end

function main()
  opt = cmd:parse(arg)

  if opt.cuda then
    require 'cutorch'
    require 'cunn'
  elseif opt.opencl then
    require 'cltorch'
    require 'clnn'
  end
  
  print("Loading dataset")
  dataset = neuralconvo.DataSet()

  print('loading ' .. opt.model_file .. '...')
  model = torch.load(opt.model_file)
  print('loaded ' .. opt.model_file)

  local in_file = io.open(opt.input_file, "r")

  local result = encode_file(in_file)
  print('saving result ' .. opt.output_file)
  torch.save(opt.output_file, result)
  print('done!')
end

main()