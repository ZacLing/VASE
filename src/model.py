import torch
import torch.nn as nn
import torch.nn.functional as F

class VASEMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation='relu', normalization=False, dropout_rate=0.0):
        super(VASEMLP, self).__init__()

        # Activation function selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky':
            self.activation = F.leaky_relu

        self.normalization = normalization

        # Hidden layers (expand -> shrink)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Expand to 2x hidden_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Shrink back to hidden_dim

        # BatchNorm layer
        self.batchnorm = nn.BatchNorm1d(output_dim)

        # Dropout layer after each Linear layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout after the first Linear layer
        x = self.activation(x)

        x = self.fc2(x)
        x = self.dropout(x)  # Apply dropout after the second Linear layer
        x = x.permute(0, 2, 1)
        if self.normalization:
            x = self.batchnorm(x)
        x = x.permute(0, 2, 1)
        return x


class VAEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 hidden_dim=256,
                 num_layers=3,
                 activation='relu',
                 normalization='batchnorm',
                 use_skip_connection=False,
                 dropout_rate=0.0):
        super(VAEEncoder, self).__init__()

        # Activation function selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky':
            self.activation = F.leaky_relu

        # Input layer without norm or skip connection
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Encoder layers (VASEMLP blocks)
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(VASEMLP(hidden_dim, hidden_dim, 2*hidden_dim, activation, True, dropout_rate))

        # Latent space parameters
        self.fc_mu = VASEMLP(hidden_dim, latent_dim, hidden_dim, activation, False, 0.0)
        self.fc_logvar = VASEMLP(hidden_dim, latent_dim, hidden_dim, activation, False, 0.0)

        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.activation(self.input_layer(x))  # Input layer, no norm or skip

        # Encoder layers with possible skip connections
        for i, layer in enumerate(self.encoder_layers):
            residual = x
            x = layer(x)
            if self.use_skip_connection:
                x = x + residual

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 input_dim,
                 hidden_dim=256,
                 num_layers=3,
                 activation='relu',
                 normalization='batchnorm',
                 use_skip_connection=False,
                 dropout_rate=0.0):
        super(VAEDecoder, self).__init__()

        # Activation function selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky':
            self.activation = F.leaky_relu

        # Input layer without skip or norm
        self.input_layer = nn.Linear(latent_dim, hidden_dim)

        # Decoder layers (VASEMLP blocks)
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(VASEMLP(hidden_dim, hidden_dim, 2*hidden_dim, activation, normalization, dropout_rate))

        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        self.use_skip_connection = use_skip_connection

    def forward(self, z):
        # z shape: (batch_size, seq_len, latent_dim)
        x = self.activation(self.input_layer(z))  # Input layer, no norm or skip

        # Decoder layers with possible skip connections
        for i, layer in enumerate(self.decoder_layers):
            residual = x
            x = layer(x)
            if self.use_skip_connection:
                x = x + residual

        out = self.decoder_out(x)
        return out


class SensorVAE(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 sensor_name=None,
                 hidden_dim=256,
                 en_layers=3,
                 de_layers=3,
                 activation='relu',
                 normalization='batchnorm',
                 use_skip_connection=False,
                 dropout_rate=0.0):
        super(SensorVAE, self).__init__()
        self.sensor_name = sensor_name
        self.encoder = VAEEncoder(input_dim=input_dim, latent_dim=latent_dim,
                                  hidden_dim=hidden_dim, num_layers=en_layers,
                                  activation=activation, normalization=normalization,
                                  use_skip_connection=use_skip_connection, dropout_rate=dropout_rate)
        self.decoder = VAEDecoder(input_dim=input_dim, latent_dim=latent_dim,
                                  hidden_dim=hidden_dim, num_layers=de_layers,
                                  activation=activation, normalization=normalization,
                                  use_skip_connection=use_skip_connection, dropout_rate=dropout_rate)


class VASE(nn.Module):
    def __init__(self, sensor_vaes, latent_dim, hidden_dim=512, num_layers=3, rnn='rnn', fuse='sum', dropout_rate=0.0):
        super(VASE, self).__init__()

        self.sensor_vaes = nn.ModuleDict(sensor_vaes)

        # RNN layer selection
        if rnn == 'rnn':
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn == 'mlp':
            self.rnn = None

        self.in_proj = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.out_proj = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

        # Fusion method (sum or mean)
        if fuse == 'sum':
            self.fuse = torch.sum
        elif fuse == 'mean':
            self.fuse = torch.mean
        elif fuse == 'prod':
            self.fuse = torch.prod

        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout to the final output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, x_dict):
        mu_dict, logvar_dict = {}, {}
        log_dict = {}
        z_list = []
        for key, x in x_dict.items():
            mu, logvar = self.sensor_vaes[key].encoder(x["data"])
            mu_dict[key], logvar_dict[key] = mu, logvar

            z = self.reparameterize(mu, logvar)
            noise = torch.randn_like(z, device=z.device)
            z = x["mask"] * z + (1 - x["mask"]) * noise
            log_dict[key] = z
            z_list.append(z)

        z = self.fuse(torch.stack(z_list, dim=0), dim=0)

        log_dict["pre_rnn"] = z
        z = self.in_proj(z)
        if self.rnn is not None:
            z, _ = self.rnn(z)  # RNN output, with no need to track hidden states
        z = self.out_proj(z)
        log_dict["post_rnn"] = z

        # z = self.dropout(z)  # Apply dropout after final projection

        # Decoding the latent vector into original space for each sensor
        out_dict = {}
        for key, model in self.sensor_vaes.items():
            out_dict[key] = model.decoder(z)

        return out_dict, mu_dict, logvar_dict, log_dict