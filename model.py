import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2,kernel_size=k_size, stride=1, padding=k_size//2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
            
        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2


class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x


class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )


    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)
        x = self.reg(x)
        return x


class net(nn.Module):
    def __init__(self, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128)
        self.cnn1 = CNN(NUM_FILTERS, FILTER_LENGTH1)
        self.cnn2 = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.reg = net_reg(NUM_FILTERS)
        self.decoder1 = decoder(FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1,FLAGS.charsmiset_size)
        self.decoder2 = decoder(FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2,FLAGS.charseqset_size)

    def forward(self, x, y, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        x_init = Variable(x.long()).cuda()
        x = self.embedding1(x_init)
        x_embedding = x.permute(0, 2, 1)
        y_init = Variable(y.long()).cuda()
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        x, mu_x, logvar_x = self.cnn1(x_embedding)
        y, mu_y, logvar_y = self.cnn2(y_embedding)
        out = self.reg(x, y).squeeze()
        x = self.decoder1(x, FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        y = self.decoder2(y, FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        return out, x, y, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y









