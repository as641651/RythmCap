local utils = {}

function utils.table_to_4Dtensor(t)
  local tensor = torch.zeros(utils.count_keys(t),t[1]:size(1),t[1]:size(2),t[1]:size(3)):type(t[1]:type())
  local idx = 1
  for k,v in pairs(t) do 
     tensor[idx] = v
     idx = idx + 1
  end
  return tensor
end

function utils.splitInput(input,feature_xdim, offset,max_clips_per_song)
  if input:size(3) > (feature_xdim + offset) then 
    local tmp = {}
    local count = 1
    while offset + feature_xdim < input:size(3) do
       tmp[count] = input:narrow(3,offset,feature_xdim)
       offset = offset + feature_xdim
       count = count + 1
       if count > max_clips_per_song then break end
    end
    return utils.table_to_4Dtensor(tmp)
  else
    return input:view(1,input:size(1),input:size(2),input:size(3))
  end
end

function utils.adjustToSize(input,xdim)
  if input:size(4) < xdim then
    local tmp = input:clone()
    input = torch.zeros(tmp:size(1),1,tmp:size(3),xdim):type(tmp:type())
    input[{{},{},{},{1,tmp:size(4)}}] = tmp
  end
  if input:size(4) > xdim then
    input = input[{{},{},{},{1,xdim}}]:contiguous()
  end
  return input
end

function utils.tensor_to_table(input)
  t = {}
  for i = 1,input:size(1) do table.insert(t,input:select(1,i)) end
  return t
end

function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

--iterates a table by sorting values in decending order
function utils.spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

return utils
