import argparse
import etl
import helpers
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
from attention_decoder import AttentionDecoderRNN
from encoder import EncoderRNN
import language

# Parse argument for language to train
# parser = argparse.ArgumentParser()
# parser.add_argument('language')
#
# # sent = 'i am a boy'
# # lang = language.Language('test')
# # lang.index_words(sent)
# # print(lang.word2index)
# # print(lang.word2count)
# # print(lang.index2word)
#
#
# args = parser.parse_args() # need to add language argument

# def MergeDataset(dataset1,dataset2):
#     en_corpus = []
#     with open(dataset1,'r') as f:
#         t = f.readline().strip()
#         en_corpus.append(t)
#         while t != '':
#             t = f.readline().strip()
#             en_corpus.append(t)
#
#     de_corpus = []
#     with open(dataset2,'r') as f:
#         t = f.readline()
#         de_corpus.append(t)
#         while t != '':
#             t = f.readline()
#             de_corpus.append(t)
#
#     de_eng = []
#     for idx, sent_en in enumerate(en_corpus):
#         sent = sent_en +'\t'+ de_corpus[idx]
#         de_eng.append(sent)
#     del de_eng[-1]
#     return de_eng
#
# de_eng = MergeDataset('./data/WMT14_en.txt','./data/WMT14_de.txt')
#
# sp_corpus = []
# with open('./data/spa-eng.txt', 'r') as f:
#     t = f.readline()
#     sp_corpus.append(t)
#     while t != '':
#         t = f.readline()
#         sp_corpus.append(t)
#
#
# with open('./data/de_eng.txt','w') as f:
#     for sent in de_eng:
#         f.write(sent)


# language = 'spa-eng'
language = 'spa-eng'
helpers.validate_language(language)

teacher_forcing_ratio = .5
clip = 5.


def train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion):
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
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_opt.step()
    decoder_opt.step()

    return loss.data / target_length

input_lang, output_lang, pairs = etl.prepare_data(language)


attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttentionDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

# Move models to GPU
# encoder.cuda()
# decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Configuring training
n_epochs = 100000
plot_every = 20
print_every = 10

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin training
for epoch in range(1, n_epochs + 1):
    # Get training data for this cycle
    training_pair = etl.variables_from_pair(random.choice(pairs), input_lang, output_lang)
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train step
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0:
        continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        time_since = helpers.time_since(start, epoch / n_epochs)
        print('%s (%d %d%%) %.4f' % (time_since, epoch, epoch / n_epochs * 100, print_loss_avg))

    if (epoch / n_epochs * 100) % 5 == 0 and epoch > 100 :
    # if epoch == 30:
        torch.save(encoder.state_dict(), './data/encoder_params_{}'.format(language))
        torch.save(decoder.state_dict(), './data/decoder_params_{}'.format(language))
        torch.save(decoder.attention.state_dict(), './data/attention_params_{}'.format(language))
        exec(open("eval.py").read())


    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0


# Save our models
torch.save(encoder.state_dict(), './data/encoder_params_{}'.format(language))
torch.save(decoder.state_dict(), './data/decoder_params_{}'.format(language))
torch.save(decoder.attention.state_dict(), './data/attention_params_{}'.format(language))

# Plot loss
helpers.show_plot(plot_losses)

def ppl(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion):
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
    # loss.backward()
    # torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    # encoder_opt.step()
    # decoder_opt.step()

    return loss.data / target_length