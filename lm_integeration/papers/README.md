# Notation

## overview

LM Integration Approaches

- shallow fusion

  - pretrained ED model and LM
  - no training integration
  - only combining scores

- deep fusion

  - pretrained ED model and LM
  - late training integration, only fine-tuning fusion components
  - tight integration by combining the hidden states

- cold fusion

  - only pretrained LM
  - early training integration, training ED model from scratch
  - tight integration

- simple fusion

- component fusion

- LM as lower decoder layer

  - pretrained ED model and LM
  - fine-tuning all parameters

- multi-task learning

  - decoder seen as a conditional LM
  - auxiliary LM task represented by a zero context vector
  - sample one of two task for optimization in each iteration

- density ratio method

  - assumption that the e2e posterior is factorizeble into an AM and LM
  - assumption of domain-invariant acoustic conditions
  - a source-domain LM trained with e2e training transcript for substraction

## INTERNAL LANGUAGE MODEL ESTIMATION FOR DOMAIN-ADAPTIVE END-TO-END SPEECH RECOGNITION

