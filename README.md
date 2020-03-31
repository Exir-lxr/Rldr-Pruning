# RldrInPruning
Restriced linear Dimensional reduction In networks Pruning. R.I.P.
- This is a data driven structure pruning lib.
- Only support to prune 1x1 conv and fc layer.

# Usage
1. Build your model and prepare corresponding checkpoint.
2. from RIP import RIPManager.
3. manager = RIPManager(structured_mode=True)
4. manager(model, (3, 224, 224))              
   (3, 224, 224) is the size of network input
5. Run your validation for 1 epoch on training set where statistic for pruning will be collected
6. manager.prune(100)
   Prune 100 channels.
- manager.prune_overview()
   you can use this function any time to check the model size and channel number.
