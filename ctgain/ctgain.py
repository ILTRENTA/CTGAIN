"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import torch.nn as nn
from ctgain.data_sampler import DataSampler

from ctgain.data_transformer import DataTransformer_with_masking_nas as DataTransformer
from ctgain.data_transformer import *

from ctgain.utils_ctgain import *


from ctgain.ctgan.synthesizers.base import BaseSynthesizer, random_state

np.random.seed(0)
torch.manual_seed(0)

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, device, pac=1, h_dim=32,hint_rate=.8):
        super(Discriminator, self).__init__()
        self._device=device
        dim = input_dim * pac #include the mask vector 
        self.pac = pac
        self.hint_rate=.7

        self.h_dim=h_dim
        self.uniform = torch.distributions.Uniform(low=0, high=1.)
        self.pacdim = dim
        seq = []
        # print(dim)
        seq+=[Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]

        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, dim), LeakyReLU(.2), Dropout(.5)]

        seq += [Linear(dim, input_dim ),
                 nn.Sigmoid()]  # before was set to 74
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_, mask):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        x_hat=input_
        
        hint = (self.uniform.sample([mask.shape[0], mask.shape[1]]) < self.hint_rate).float().to(self._device)
        hint = mask * hint ## the size of the hint matrix is wrong 
        
        
        inp = torch.cat([x_hat, hint], dim=1)
        
        # print( "#### 78 ####line ----->",inp.view(-1, self.pacdim).shape)
        
        return self.seq(inp)#.view(-1, self.pacdim))
        #return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_, pred=False):
        """Apply the Residual layer to the `input_`."""
        
        # print(1)
        if pred:
            print(input_.dtype)
            input_=input_.to(torch.float64)

        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        # print(" output ", out.shape, " input ",input_.shape)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, device):
        super(Generator, self).__init__()
        self._device=device
        self.seed_sampler = torch.distributions.Uniform(low=0, high=0.01)
        self.h_dim = embedding_dim
        # print("line 99 --- > embedding dims: ", embedding_dim)

        dim = data_dim # include the mask vector

        
        # seq = [Residual(dim, data_dim*2)]

        i=0
        # for item in list(generator_dim):
        #     seq += [Residual(dim , item)]
        #     print( dim , item)
        #     dim += item
        dim
        ## beginning trapianto 

        generator = [Residual(dim * 2, dim)]
        # generator = [Residual(dim, dim)]
        # generator = [Residual(dim , dim)]
        
        generator.extend([Residual(dim+dim+dim, dim)])
        generator.extend([Residual(dim*4, dim)])

        generator.extend([Linear( dim*5,  dim)])

        for layer in generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.seq = Sequential(*generator)


        # ## end trapianto
        # seq+=[Residual(dim, data_dim)]

        # # i+=1
        
        # seq.append(Linear(dim, data_dim))

        # print( "line 117 eccolo ", (dim , data_dim))
        # print(i)
        # self.seq = Sequential(*seq)

    def forward(self, input_, mask, pred=False):
        """Apply the Generator to the `input_`."""
        data_norm=input_

        data_norm[mask==0]=0
        # mask=torch.from_numpy(mask).to(self._device)
        # print("line 114 ---> data norm shape: ", data_norm.shape)
        z = self.seed_sampler.sample([data_norm.shape[0], data_norm.shape[1]]).to(self._device)
        # z = self.seed_sampler.sample([data_norm.shape[0], self.h_dim]).to(self._device)
        
        # print("line 116 ----> z shape: ", z.shape)
        # print( type(mask), type(z))
        
        # mask=torch.from_numpy(mask).to(self._device)
        # if mask.shape[0] != z.shape[0]:
        #     mask=mask.repeat(z.shape[0], 1)
        #     print( mask.shape, z.shape)
        # if type(mask)==np.ndarray:
        #     mask=torch.from_numpy(mask).to(self._device)                     
        z=z.to(self._device)
        mask=mask.to(self._device)
        data_norm.to(self._device)
        random_combined = mask * data_norm + (1-mask) * z
        # if type(random_combined)==np.ndarray:
        #     print( type(random_combined), type(mask))
        #     random_combined=torch.from_numpy(random_combined).to(self._device)
        #     mask=torch.from_numpy(mask).to(self._device)
        # if random_combined.dtype!= mask.dtype:
        #     random_combined=random_combined.type(mask.dtype)

      

        xoxo=torch.cat([random_combined, mask], dim=1)
        if pred:

            sample = self.seq(xoxo)
        
        else: 
            sample = self.seq(xoxo)
        x_hat = random_combined * mask + sample * (1-mask)
        # data = self.seq(input_)

        return sample, random_combined, x_hat





class CTGAIN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac= 2, 
                 alpha=.8, hint_rate=.8, cuda=True):

        assert batch_size % 2 == 0

        self.alpha=alpha
        self.hint_rate=hint_rate

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda' ######################################################### HERE

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        from tqdm import tqdm
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data, mask = self._transformer.transform(train_data)

        train_data, mask= torch.from_numpy(train_data).float().to(self._device), torch.from_numpy(mask).float().to(self._device)

        self._data_sampler = DataSampler(
            train_data, mask, 

            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        # print("line 363 ---> data_dim ", data_dim)

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim, self._device
        ).to(self._device)

        
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim, self._device,
            pac=self.pac, hint_rate=self.hint_rate
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        first=True
        for i in tqdm(range(epochs)):
            
            for id_ in range(steps_per_epoch):
                
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real, mask_samp = self._data_sampler.sample_data(self._batch_size, col, opt)
                        real, mask_samp= real.to(self._device),\
                            mask_samp.to(self._device)
                        
                        # print("cond 1 ")
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real, mask_samp = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                        
                        real, mask_samp=real.to(self._device), mask_samp.to(self._device)
                        # print("cond 2")
                        # real, mask_samp= torch.from_numpy(real.astype('float32')).to(self._device),\
                        #     torch.from_numpy(mask_samp).to(self._device)
                    # fake = self._generator(fakez, mask_samp) # for now 
                    # fakeact = self._apply_activate(fake)
                    ## other implementation

                    # real = torch.from_numpy(real.astype('float32')).to(self._device)
                    # mask_samp=torch.from_numpy(mask_samp).to(self._device)

                    sample, random_combined, x_hat = self._generator(real, mask_samp) 
                    fakeact = self._apply_activate(x_hat)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact
                    
                    # y_fake = discriminator(fake_cat)

                    d_prob = discriminator(real_cat, mask_samp)

                    # pen = discriminator.calc_gradient_penalty(
                    #     real_cat, fake_cat, self._device, self.pac)
                    # print("dprob --- >",d_prob.shape)
                    # print("mask_samp ---->", mask_samp.shape)
                    # if first:
                    #     print("real_cat ---->", real_cat)
                    #     print("fake_cat ---->", fake_cat)
                    #     print("d_prob ---->", d_prob)
                    #     print("mask_samp ---->", mask_samp)
                    #     print( torch.log(d_prob+1e-7))
                    #     first=False
                    #     print( "d_prob ---->", d_prob.shape)

                    #     print( "mask_samp ---->", mask_samp.shape)
                    loss_d = -torch.mean(mask_samp*torch.log(d_prob+1e-8) + (1-mask_samp)*torch.log(1-d_prob + 1e-8))
                    
                    # loss_d = -torch.mean(mask_samp*torch.log(d_prob+1e-8) - (1-mask_samp)*torch.log(1-d_prob + 1e-8))
                    
                    optimizerD.zero_grad()
                    # pen.backward(retain_graph=True) # check if penalty is necessary 
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                # if condvec is None:
                #     c1, m1, col, opt = None, None, None, None
                # else:
                #     c1, m1, col, opt = condvec
                #     c1 = torch.from_numpy(c1).to(self._device)
                #     m1 = torch.from_numpy(m1).to(self._device)
                #     fakez = torch.cat([fakez, c1], dim=1)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real, mask_samp = self._data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real, mask_samp = self._data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm])
                    c2 = c1[perm]


                sample, random_combined, x_hat = self._generator(real, mask_samp)

                # sample, random_combined, x_hat=torch.from_numpy(sample).to(self._device),\
                #       torch.from_numpy(random_combined).to(self._device),\
                #           torch.from_numpy(x_hat).to(self._device)
                
                fakeact = self._apply_activate(x_hat)
                
                if c1 is not None:
                    d_prob = discriminator(torch.cat([fakeact, c1], dim=1),
                                            mask=mask_samp)
                else:
                    d_prob = discriminator(fakeact,
                                            mask=mask_samp)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                # g_loss = -torch.mean((1-mask_samp) * torch.log(d_prob + 1e-8))
                
                g_loss = -torch.mean((1-mask_samp) * torch.log(d_prob + 1e-8))
                
                mse_loss = torch.mean(torch.pow((mask_samp * random_combined - mask_samp*sample), 2)) / torch.mean(mask_samp)  
                
                
                generator_loss = g_loss + self.alpha * mse_loss

                optimizerG.zero_grad()
                generator_loss.backward()
                optimizerG.step()

            if self._verbose:
                print(f'Epoch {i+1}, Loss G: {generator_loss.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)
    @random_state
    def impute(self, incomp_data, real_only=True):#condition_column=None, condition_value=None
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        # if condition_column is not None and condition_value is not None:
        #     condition_info = self._transformer.convert_column_name_value_to_id(
        #         condition_column, condition_value)
        #     global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
        #         condition_info, self._batch_size)
        # else:
        #     global_condition_vec = None

        dta, mask_samp=self._transformer.transform(incomp_data)
        n=dta.shape[0]
        dta, mask_samp= torch.from_numpy(dta).\
            to(self._device), torch.from_numpy(mask_samp).to(self._device)
        
        # steps = n // self._batch_size + 1
        # data = []
        # for i in range(steps):
        #     # mean = torch.zeros(self._batch_size, self._embedding_dim)
        #     # std = mean + 1
        #     # fakez = torch.normal(mean=mean, std=std).to(self._device)
            
        # mask_samp=torch.from_numpy(mask_samp).to(torch.float64).to(self._device)
        # print(mask_samp.shape, dta.shape)
        # print(dta.dtype, mask_samp.dtype)
        sample, random_combined, x_hat = self._generator(dta, mask_samp, pred=True)
        fakeact = self._apply_activate(x_hat)

        # data.append(fakeact.detach().cpu().numpy())
        data=fakeact.detach().cpu().numpy()
        # print("data before transformation----- >",data)
        # data = np.concatenate(data, axis=0)
        # data = data[:n]

        if real_only:
            dta=self._transformer.inverse_transform(data)
            
            
            #dta=incomp_data+incomp_data.isnull().astype(int)*dta
            
            for col in incomp_data.columns:
                # print(col)
                incomp_data.loc[incomp_data[col].isna(), col]=dta.loc[incomp_data[col].isna(),
                                                                 col]
            

            return incomp_data
        else:
            return self._transformer.inverse_transform(data)




    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
    
    # def discriminator_loss(self, mask, d_prob):
    #     d_loss = -torch.mean(mask * torch.log(d_prob+1e-8) + (1-mask) * torch.log(1-d_prob + 1e-8))
    #     return d_loss

    # def generator_loss(self, mask, d_prob, random_combined, sample):
    #     g_loss = -torch.mean((1-mask) * torch.log(d_prob + 1e-8))
    #     mse_loss = torch.mean(torch.pow((mask * random_combined - mask*sample), 2)) / torch.mean(mask)
    #     return g_loss, mse_loss