# -*- coding: utf-8 -*-

import argparse
import etl
import helpers
import torch
from attention_decoder import AttentionDecoderRNN
from encoder import EncoderRNN
from language import Language
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import random
# Parse argument for input sentence
# parser = argparse.ArgumentParser()
# parser.add_argument('language')
# parser.add_argument('input')
# args = parser.parse_args()
language = 'spa-eng'
helpers.validate_language_params(language)

input_lang, output_lang, pairs = etl.prepare_data(language)








attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
teacher_forcing_ratio = .5
clip = 5.


def GetLoss(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion):
    # Initialize optimizers and loss
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0

    # Get input and target seq lengths
    target_length = target_var.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([0]))
    # decoder_input = decoder_input.cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    # decoder_context = decoder_context.cuda()
    decoder_hidden = encoder_hidden

    # Scheduled sampling
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Feed target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_var[di])
            decoder_input = target_var[di]
    else:
        # Use previous prediction as next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)

            loss += criterion(decoder_output, target_var[di])


            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input

            if ni == 1:
                break

    # Backpropagation


    return loss.data / target_length

import math

def evaluate_ppl(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,idx):
    loss = GetLoss(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    ppl_score = min(math.exp(loss),100)
    if idx % 100 == 0:
        print('ppl',idx,'example score:',ppl_score)
    return ppl_score


def ppl_loop(input_list, output_list, n_sample):
    sum_loss = 0
    for idx in range(n_sample):
        idx += 20000
        inputs = input_list[idx]
        outputs = output_list[idx]
        input_sentence = helpers.normalize_string(inputs)
        output_sentence = helpers.normalize_string(outputs)
        pairs = [input_sentence, output_sentence]
        training_pair = etl.variables_from_pair(pairs, input_lang, output_lang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        loss = evaluate_ppl(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                            criterion,idx)
        sum_loss += loss
    print('final ppl score:',sum_loss/n_sample)



ppl_loop(input_list,output_list,10)










def MakeNgram(corpus, n):
    data = []
    corpus = corpus.split(' ')
    for i in range(len(corpus)-(n-1)):
        data.append(corpus[i: i+n])
    return data


import numpy as np


def GetBleuScore(input_data, output_list, ngram):
    total_score = 0
    sent_count=0
    for index, token in enumerate(input_data):
        if index == 200:
            break
        inputs = token[0:-1]
        sentence = helpers.normalize_string(inputs)
        output_words, decoder_attn = evaluate(sentence)
        output_words = output_words[0:-2]
        output_sentence_raw = ' '.join(output_words)
        answer_sentence_raw = output_list[index].lower()[0:-1]
        # print(index+1,'번째')
        # print('1-gram')
        # print(MakeNgram(output_sentence,1))
        # print(MakeNgram(answer_sentence,1))
        # print('2-gram')
        # print(MakeNgram(output_sentence,2))
        # print(MakeNgram(answer_sentence,2))
        # for n in ngram:
        output_sentence = MakeNgram(output_sentence_raw, 1)
        answer_sentence = MakeNgram(answer_sentence_raw, 1)
        if len(output_sentence) <= 4 or len(answer_sentence) <= 4 :
            continue
        A = min([1, len(output_sentence) / len(answer_sentence)])
        score_list = []
        for n in range(1,ngram+1):
            count = 0
            output_sentence = MakeNgram(output_sentence_raw,n)
            answer_sentence = MakeNgram(answer_sentence_raw,n)
            for output_word in output_sentence:
                if output_word in answer_sentence:
                    count += 1
            b = count / len(output_sentence)
            score_list.append(b)
            if n==ngram:
                B = np.prod(score_list, axis=0)**(1/n)
                score = A*B
                total_score += score
        sent_count += 1
    print(sent_count)
    if sent_count == 0 :
        sent_count = 1
    total_score = total_score/sent_count
    return total_score*100

    # return voca_score


input_list = input_list[80000::]
output_list = output_list[80000::]
result = GetBleuScore(input_list, output_list, 1)
print(result)


