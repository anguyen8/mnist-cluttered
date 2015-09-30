
require 'torch'

-- local train = torch.load("train_32x32.t7", 'ascii')
-- print(train)

-- local test = torch.load("test_32x32.t7", 'ascii')
-- print(test)


function convert(train)
  -- convert data
  local data = train.data
  local size = data:size(1)   -- number of images

  local labels = train.labels

  -- new containers
  local new_data = torch.Tensor(size, 1, 28, 28)
  local new_labels = torch.Tensor(size)

  for i=1, size do

    -- data
    new_data[i] = torch.reshape(data[i], torch.LongStorage{1, 28, 28})

    -- label
    new_labels[i] = labels[i][1]
    
    -- print(img:size())
  end

  -- convert labels

  -- print(labels:size())

  train.data = new_data
  train.labels = new_labels

  -- print (train)

  return train
end

function convert_train_valid_to_train()
  local load_train = torch.load("train.t7")
  local load_val = torch.load("valid.t7")

  local save_train = convert(load_train)
  local save_val = convert(load_val)

  local save_data = torch.cat(save_train.data, save_val.data, 1)
  local save_labels = torch.cat(save_train.labels, save_val.labels, 1)

  print("----------------------")
  -- print(save_data:size())
  -- print(save_labels:size())

  local final_train = {
    data = save_data,
    labels = save_labels
  }

  print(final_train)

  torch.save("train_28x28.t7", final_train)
end

function convert_test()
  local load_test = torch.load("test.t7")
  local final_test = convert(load_test)
  torch.save("test_28x28.t7", final_test)
  print(final_test)
end





