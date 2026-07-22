# Guia de estudo — detecção de anomalia em imagens

Diário anotado dos conceitos por trás deste laboratório. A ideia é ler isto de uma
sentada para entender *por que* cada peça existe, e depois voltar aos módulos do
`src/` (todos comentados) e ao notebook de lacunas para fixar na prática. O idioma
do código é inglês (padrão do projeto); este guia é o caderno em português.

## 1. O problema, e por que ele é ao contrário do usual

Inspeção de peça — desgaste, trinca, risco, amassado — tem um formato ingrato:

- exemplos **bons** existem aos montes (toda peça aprovada é um);
- **falhas** são raras e, pior, cada uma é diferente da outra.

Montar um conjunto rotulado com "todos os tipos de defeito" é inviável — o próximo
defeito nunca visto sempre aparece. A virada de chave é parar de tentar *classificar
defeitos* e passar a **modelar o normal**: aprende-se como é uma peça boa e trata-se
qualquer desvio como candidato a anomalia. Isso é *detecção de anomalia não
supervisionada*, e é a mesma família de ideias de modelos médicos que aprendem
tecido saudável e sinalizam o resto.

Como não existe dataset aberto de peça automotiva desgastada, usamos o benchmark
padrão da área — **MVTec AD** — começando por `metal_nut` e `screw`, as categorias
metálicas mais próximas do alvo, e que vêm com **máscaras por pixel** do defeito
(dá para medir *onde* o modelo olha, não só *se* ele acerta).

## 2. O protocolo do MVTec (é ele que amarra tudo)

```
<categoria>/
  train/good/              -> SÓ imagens boas. É tudo que se usa para "treinar".
  test/good/               -> boas de teste (devem pontuar BAIXO)
  test/<tipo_de_defeito>/  -> risco, amassado... (devem pontuar ALTO)
  ground_truth/<tipo>/     -> máscara por pixel do defeito
```

A disciplina "treino = só boas" é sagrada: mostrar um defeito no treino é trapaça e
destrói o sentido do método. No código, isso é imposto num único lugar
(`list_samples` em `src/image_anomaly_lab/data.py`), justamente para não vazar.

## 3. Dois métodos, de propósito em contraste

Espelhando o `mnist-metric-lab` (dois jeitos do mesmo problema), aqui há um método
fraco e um forte, e a graça está em ver a diferença.

### 3.1 Baseline — autoencoder de reconstrução (o fraco)

Treina-se uma rede a comprimir uma imagem boa num vetor pequeno e reconstruí-la,
minimizando o erro **só em imagens boas**. Aposta: ela fica boa em reconstruir
textura *normal* e ruim no que nunca viu, então erro alto de reconstrução = defeito.

**Por que falha (essa é a lição):** autoencoders convolucionais generalizam *demais*.
Um risco é, localmente, "mais umas bordas", e uma rede que aprendeu a reconstruir
todas as bordas de uma peça boa reconstrói o risco também — o erro fica baixo
justamente onde precisava ficar alto. Ajustar o gargalo (`latent_dim`) não resolve:
é um fio da navalha, não tem tamanho mágico.

### 3.2 Método forte — banco de memória (PatchCore-lite)

Aqui mora a ponte com o seu triplet loss. Dois passos:

- **Fit:** passa cada imagem boa por uma CNN **pré-treinada e congelada** (backbone
  da ImageNet), coleta um *embedding* por região ("patch") e joga todos num único
  **banco de memória** do que é normal. Não há gradiente, não há treino — só memória.
- **Score:** para uma imagem de teste, embeda os patches do mesmo jeito e pergunta,
  para cada um, *qual a distância do meu vizinho normal mais próximo?* Patch limpo
  cai em cima de uma entrada do banco (distância minúscula); patch riscado não tem
  vizinho próximo (distância grande). Essas distâncias, remontadas na grade da
  imagem, **são o heatmap**; o score da imagem é a maior distância entre os patches.

É *embedding + busca do vizinho mais próximo* — a mesma maquinaria de metric
learning, agora apontada para defeitos. Quem fez triplet loss já sabe o núcleo.

**Por que funciona sem treinar nada:** as features intermediárias de uma CNN da
ImageNet já descrevem conteúdo local de forma rica e geral. Um patch de metal
escovado limpo e um com risco produzem vetores diferentes — e essa diferença é tudo
que o vizinho-mais-próximo precisa.

## 4. Detalhes que parecem pequenos e não são

- **Normalização ImageNet (`data.py`).** O backbone foi treinado com imagens
  padronizadas pela média/desvio da ImageNet; as features só fazem sentido se a
  entrada for padronizada igual. Mesmo transform no treino e no teste — *paridade
  treino/serve*: qualquer diferença aqui desloca os embeddings de teste em relação ao
  banco e arruína as distâncias.
- **Alinhar as camadas (`backbones.py`).** Camadas mais profundas têm grade espacial
  mais grossa. Para empilhar `layer2` e `layer3` num vetor por patch, sobe-se a mais
  profunda para a grade da mais rasa *antes* de concatenar. Errar isso cola canais de
  locais diferentes — silenciosamente, sem erro, só piorando o resultado.
- **`max` sobre os patches (`memory_bank.py`).** O score da imagem é o pico do mapa,
  não a média: uma peça com um único risco pequeno ainda é defeituosa, e a média
  afogaria esse pico no mar de pixels normais.
- **Suavizar o mapa.** Um patch alto isolado costuma ser ruído; um defeito real
  acende uma vizinhança. O blur gaussiano codifica essa crença.
- **Coreset.** Guardar todo patch de toda imagem boa é redundante (patches vizinhos
  são quase idênticos). Guarda-se uma fração (`coreset_ratio`). A acurácia quase não
  cai até frações bem pequenas — veja o experimento 3 do `STUDY_GUIDE.md`.

## 5. Como se mede (e as armadilhas de cada métrica)

- **Image AUROC** — separa boa de defeituosa no nível da imagem. Livre de threshold;
  é a métrica-título de *detecção*.
- **Pixel AUROC** — trata cada pixel como amostra. Recompensa heatmap que acende no
  defeito e fica quieto no resto. *Armadilha:* defeito é uma fração minúscula dos
  pixels, então um mapa preguiçoso pontua alto só por ser calmo no fundo normal.
- **PRO (Per-Region Overlap)** — corrige a armadilha acima pontuando cada *região*
  conexa de defeito por igual (um risco fininho vale tanto quanto um amassão) e
  integrando a cobertura contra a taxa de falso-positivo até FPR=0.3. Complementa o
  pixel AUROC.
- **Threshold.** *Youden's J* maximiza TPR−FPR, mas precisa de defeitos rotulados —
  que muitas vezes não se tem. A regra do *percentil das boas* ("sinalize acima do
  percentil 99 das boas conhecidas") não precisa de defeito nenhum e é a que de fato
  se implanta numa linha. Veja o experimento 5 do `STUDY_GUIDE.md`.

## 6. Reprodutibilidade e hardware

Todo número gravado (`results.py`) vem carimbado com a configuração e com o runtime
do torch (`devices.py`). Na placa AMD RDNA4, o build ROCm do PyTorch se reporta pela
mesma API `torch.cuda` de uma placa NVIDIA — o que distingue é `torch.version.hip`.
Por isso `describe_torch` guarda essa informação: um "AUROC=0.98" sem saber o
hardware que o produziu não é reprodutível.

## 7. Próximos passos (ficam de exercício)

- Coreset guloso (*farthest-point*) no lugar da subamostragem aleatória.
- PaDiM (gaussiana por posição + distância de Mahalanobis) como terceiro detector.
- Varrer as 15 categorias do MVTec e montar uma tabela geral.
