# NER
Sequence tagging task based on [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) (ACL 2016) paper .

## Dataset
[Few-NERD](https://arxiv.org/abs/2105.07464) is a large-scale, fine-grained manually annotated named entity recognition dataset, which contains 8 coarse-grained types, 66 fine-grained types, 188,200 sentences, 491,711 entities and 4,601,223 tokens. Three benchmark tasks are built, one is supervised: Few-NERD (SUP) and the other two are few-shot: Few-NERD (INTRA) and Few-NERD (INTER).

## Model
The neural network architecture uses word- and character-level representations. The CNN computes character representations for each word, capturing morphological details. Then character-level representations are concatenated with word embeddings and fed into the BiLSTM layer. In sequence labeling, it is important to have both past and future context. The BiLSTM considers both directions, combining past and future information separately. Finally, the combination of these two hidden states of BiLSTM goes to the CRF layer to decode the best label sequence, considering correlations between labels in neighborhoods.

<p align="left">
  <img src="img/cnn_bilstm_crf.png" width="100%" />
</p>

<!-- ## Output
Sentences generated by ChatGPT:

- <div style='font-size:14px'>The <span style='color:#1f77b4'>Mona Lisa, </span> <span style='color:grey'>(art-other)</span> painted by <span style='color:#e377c2'>Leonardo da Vinci, </span> <span style='color:grey'>(person-other)</span> is displayed in the <span style='color:#ff7f0e'>Louvre Museum, </span> <span style='color:grey'>(building-other)</span> a renowned cultural institution in Paris. </div>

- <div style='font-size:14px'>The <span style='color:#7f7f7f'>SpaceX Falcon 9 rocket, </span> <span style='color:grey'>(product-other)</span> designed by <span style='color:#9467bd'>Elon Musk's aerospace company, </span> <span style='color:grey'>(organization-other)</span> successfully launched a payload into orbit. </div>

- <div style='font-size:14px'>The 2018 <span style='color:#2ca02c'>Winter Olympic Games </span> <span style='color:grey'>(event-sportsevent)</span> witnessed the triumphs of athletes like <span style='color:#e377c2'>Chloe Kim </span> <span style='color:grey'>(person-other)</span> and <span style='color:#e377c2'>Johannes Høsflot Klæbo, </span> <span style='color:grey'>(person-other)</span> becoming household names. </div>

- <div style='font-size:14'>In the <span style='color:#ff7f0e'>Louvre </span> <span style='color:grey'>(building-other)</span> museum, visitors admired the exquisite paintings of <span style='color:#e377c2'>Vincent van Gogh </span> <span style='color:grey'>(person-artist/author)</span> while discussing the impact of his work on modern art. </div>

- <div style='font-size:14px'>The <span style='color:#8c564b'>Nobel Prize </span> <span style='color:grey'>(other-award)</span> honors remarkable contributions in different fields. Established by <span style='color:#e377c2'>Alfred Nobel's </span> <span style='color:grey'>(person-artist/author)</span> will, the prizes are given annually to individuals and organizations that have made outstanding advancements for the benefit of humanity. </div>

- <div style='font-size:14px'>Scientists at <span style='color:#9467bd'>NASA </span> <span style='color:grey'>(organization-government/governmentagency)</span> recently discovered a new exoplanet in a distant galaxy, using advanced telescopes and data analysis techniques. </div>

- <div style='font-size:14px'>The <span style='color:#9467bd'>FIFA World Cup </span> <span style='color:grey'>(organization-sportsleague)</span> captivated global audiences with thrilling soccer matches. </div>

- <div style='font-size:14px'>The <span style='color:#ff7f0e'>Great Barrier Reef, </span> <span style='color:grey'>(building-other)</span> located off the coast of <span style='color:#d62728'>Australia, </span> <span style='color:grey'>(location-GPE)</span> is a <span style='color:#9467bd'>UNESCO World Heritage </span> <span style='color:grey'>(organization-other)</span> site and home to a diverse marine ecosystem. </div>

- <div style='font-size:14px'>The <span style='color:#8c564b'>Hubble Space Telescope, </span> <span style='color:grey'>(other-astronomything)</span> a marvel of astronomy <span style='color:#8c564b'>technology, captures breathtaking </span> <span style='color:grey'>(other-astronomything)</span> images of distant galaxies and nebulae. </div>

- <div style='font-size:14px'>The <span style='color:#2ca02c'>European Under-19 Football Championship </span> <span style='color:grey'>(event-sportsevent)</span> featured emerging talents like <span style='color:#e377c2'>Luca Hernandez </span> <span style='color:grey'>(person-other)</span> and <span style='color:#e377c2'>Sophia Müller, </span> <span style='color:grey'>(person-other)</span> showcasing the future of soccer. </div> -->

## Reference
- [FEW-NERD: A Few-shot Named Entity Recognition Dataset](https://arxiv.org/abs/2105.07464) (ACL-IJCNLP 2021)
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://dl.acm.org/doi/10.5555/645530.655813) (ICML 2001)
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) (2015)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) (ACL 2016)