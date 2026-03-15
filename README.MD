# Few dimensions drive the modality gap
All the pretrained encoder from contrastive models, exhibit modality gap.These models might be very effective for retrieval tasks, which is the most important task for multimodal learning but the latent space is usually fragmented due to the modality gap and cannot be used directly for other task such as clustering or classification, in which the key requirements is to have semantic groups in the latent space. 
<br><br>
Training from scratch these encoders to remove the modality gap is very expensinve in terms of computational power and data requirements.
<br><br>
We want a method that we can apply to the pretrained encoder that allows us to get the same performace in retrieval while having a latent space more homogeneous and less fragmentedsuch that we can perform other tasks rather then the only retrieval.


## Todo:
- classification analysis for the dasets (cifar10, mscoco with imagenet labels, flicker30k with imagenet labels, msrvtt with its labels), 
FiLMR layer in file analysis/filmr_scott.py
