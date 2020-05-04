
library(tibble)
library(readr)
library(keras)

data_dir <- "C:/Users/moham/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

glimpse(data)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()


#Preparing the data
data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size
# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

#mean(abs(preds - targets))

evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}


# Using checkpoints and Tensorboard commands
#Train the model and pass it the callback_model_checkpoint:
checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  period = 5,
  verbose = 1
)

model <- create_model()


# launch TensorBoard (data won't show up until after the first epoch)
tensorboard("logs/run_a")

model <- keras_model_sequential() %>% 
  layer_lstm(units = 16, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_lstm(units = 32, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 15,
  validation_data = val_gen,
  validation_steps = val_steps,
  callbacks = list(cp_callback),  # pass callback 'checkpoints' to training   
  callbacks = callback_tensorboard("logs/run_a")# pass callback 'tensorboard' to training
)

plot(history)

# another model
model_1 <- keras_model_sequential() %>% 
  layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
             input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)
model_1 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history_1 <- model_1 %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 10,
  validation_data = val_gen,
  validation_steps = val_steps,
  callbacks = list(cp_callback),  # pass callback 'checkpoints' to training
  callbacks = callback_tensorboard(log_dir = "logs/run_b", histogram_freq = 2)# pass callback 'tensorboard' to training with histogram
  
)

plot(history_1)

tensorboard("logs")