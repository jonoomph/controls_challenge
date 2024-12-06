# Training Notes

## Run Game & Collect Data
Before training the model, first run the game and collect data for all the levels
you want to include in your training run. Data is stored in the `game/data` folder.
The `levels.json` and `high-score.json` files are also located in this folder.

```
python3 run_game.py
```

## Optimize Data (Optional)
Some game data can be further optimized by running the optimize data script. This
iterates through the `game/data` folder, and finds any files not yet optimized (i.e. in the `game/data-optimized` folder).
We then iterate through each steer value +/- a few steering adjustments to see if a better
score results. WARNING: If we optimize TOO MUCH, it will hurt performance, since it will 
be essentially finding the noise pattern from the TinyPhysics simulator.

```
python3 optimize_simulations.py
```

## Prepare Training Data
Once you have some human recorded game data saved, it's time to prepare the simulations
for our training run. **NOTE:** Edit `save_simulations.py` if you want to include additional
level #'s for the PID dataset. You can find new level candidates by running `plot_controllers.py` or 
`plot_levels.py`.

In other words, in order to combine human game replay data with other PID controllers, edit
this script to include additional levels (or additional PID controllers). The controller with the
best score is used to generate the final training data simulations.

```
python save_simulations.py
```

This will populate the `game/train/simulations` folder, and because it chooses the data from the 
controller with the best/lowest score, you will end up with a distrubtion of training data from different
controllers like this:

```
PID-TOP: 58 wins
PID-EXPERIMENTAL: 50 wins
PID-REPLAY: 68 wins
PID-FF: 70 wins
PID: 5 wins
PID-FUTURE: 1 wins
```

## Train the Model

To train the model, run the following script. This will iterate through
each file in the `game/train/simulations` folder, collect the input window of input states, and train
the neural network. You can adjust the hyperparameters by editing the `train.py` script, or
passing in args to overwrite them.

```
python3 train.py
```

For graphing of the training run, I am using TensorBoard. To run the localhost server:

```
 tensorboard --logdir=runs
```