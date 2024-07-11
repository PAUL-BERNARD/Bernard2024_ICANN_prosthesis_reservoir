# Prediction of reaching movements with target information towards trans-humeral prosthesis control using Reservoir Computing and LSTMs

### Abstract

Controlling a prosthetic upper limb requires the reconstruction of multiple distal articulations. Moreover, the higher the amputation level, the more joints need to be reconstructed, and the less kinetic information is available in the residual limb. By exploiting contextual information, such as the position and orientation of a target in a reaching task, we aim to reconstruct the natural dynamics of the distal joints using recurrent neural networks. We compare  performances of two models, an Echo State Network (ESN) and an LSTM, on two conditions: training on individual subjects, and training on a 5-fold CV on 15 subjects. 
We explored hyperparameters on both models: the ESN shows better performances on the single-subject task, and the LSTM shows better performances on the multiple-subject task.
When looking qualitatively at the predictions, we observe that even if networks don't have the same MSE errors, they perform the task well and are able to reach the targets most of the time. We further analyze the performance of the models on the multi-subject task and report different kinds of generalizations.

### Supplementary materials

Supplementary materials can be found in the `supplementary/` folder.