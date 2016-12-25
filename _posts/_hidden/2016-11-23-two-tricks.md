---
layout: post
title: Ladder VAE, Bits-Back Coding, and Conjugacy
disqus: y
---

It's been a while since the [Ladder Variational Autoencoder](https://arxiv.org/abs/1602.02282) came out, but I think it is a paper worth revisiting, especially given the ever growing interest in Deep Latent Gaussian Models (DLGMs).
The paper proposed multiple orthogonal methods to tackle the optimization issues associated with DLGMs. But for the purposes of this post, I wish to focus on two particular tricks.

The two tricks I am referring to are *top-down inference* and *precision-weighted merging*. I will motivate these two tricks in order and explain why I find them so interesting.

---

### Inference in DLGMs with bottom-up inference

In a typical DLGM, we have a hierarhical data-generating mechanism

$$
\begin{align}
p_\theta(z_{1:T}, x) &= p(z_T) \left(\prod_{i < T} p_\theta(z_i \mid z_{i+1}) \right) p_\theta(x \mid z_1),
\end{align}
$$

where

$$
\begin{align}
z_i &\sim p_\theta(z_i \mid z_{i+1}) 
= \mathcal{N}(z_i \mid \mu_{\theta, z_i}(z_{i+1}), \sigma^2_{\theta, z_i}(z_{i+1})) \\
x &\sim p_\theta(x \mid z_1) =  \mathcal{B}(x \mid \mu_{\theta, x}(z_1)).
\end{align}
$$

To train the generative model parameter \\(\theta\\), we will use per-sample amortized inference. Our goal is to find a good variational approximation \\( q\_\phi(z\_{1:T} \mid x) \\) of the true posterior \\(p\_\theta(z\_{1:T} \mid x)\\) (which we do not have access to). Naively, we could use mean-field approximation, but this can lead to a very bad variational approximation, which in turn leads to a bad generative model. Since each \\(z\_i\\) *d*-separates \\(z\_{>i}\\) and \\((x, z\_{<i})\\), a smart (and natural) approach would be to do bottom-up inference, where

$$
\begin{align}
q_\phi(z_{1:T} \mid x) &= q_\phi(z_1 \mid x) \prod_{i > 1} q_\phi(z_{i+1} \mid z_i).
\end{align}
$$

While this leads to better performance than mean-field, in practice, not all of the stochastic layers end up being used in the model. In other words, up to some \\(i\\), the mutual information between \\(z_{>i}\\) and \\(x\\) becomes zero. 

### Drawbacks of bottom-up inference

There are a couple of ways I can think of for why bottom-up inference underutilize the stochastic layers. From an optimization point-of-view, bottom-up inference suffers from the broken-telephone effect. With each stochastic layer, it becomes increasingly difficult to accurately propagate the information about \\(x\\). Consequently, most of the higher-level layers end up unused.

Another--perhaps more interesting--reason has to do with the behavior of the bits-back coding mechanism. I learned about this from the [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) paper. The main idea is that variational approaches to learning generative models tries as much as possible to *not* use the latent variables if the choice of variational family is poor (i.e. the KL-divergence between \\(q\\) and \\(p\\) is high when \\(z\\) is useful). Keeping this idea in mind, recall that our variational family is the set of Gaussians, it is easy to see why \\(q\_\phi(z\_{i+1} \mid z\_i)\\) has little chance of matching \\(p\_\theta(z\_{i+1} \mid z\_i)\\). The model is thus reluctant to use \\(z_1\\), doubly reluctant to use \\(z_2\\), and so on.

### Top-down inference to the rescue

In top-down inference, we simply flip the order of inference. Rather than following the natural chain of inferring our way up the stochastic layers, we defy all logic and do the exact opposite

$$
\begin{align}
q_\phi(z_{1:T} \mid x) &= q_\phi(z_T \mid x) \prod_{i < T} q_\phi(z_i \mid z_{i+1}, x).
\end{align}
$$

It is easy to see how this solves the broken-telephone issue: all of our latent variables are now directly conditioned on \\(x\\) during inference! To also see how this addresses the bits-back coding behavior, note that in top-down inference, each \\(q\_\phi(z_i \mid z_{< i}, x)\\) is implicitly defined as

$$
\begin{align}
q_\phi(z_{i+1} \mid z_i) = 
\end{align}
$$

---

### Top-down inference is awkward

The one redeeming quality of bottom-up inference is that inference implementation is straight-forward. For the most part, all you need are neural networks \\((\mu\_{\phi, z\_{i+1}}, \sigma^2\_{\phi, z\_{i+1}})\\) that take \\(z_i\\) as input and outputs the distribution of \\(z_{i+1}\\). In top-down inference, however, the neural networks \\((\mu\_{\phi, z\_i}, \sigma^2\_{\phi, z\_i})\\) need to take both \\(z\_{i+1}\\) and \\(x\\) as inputs. This can get awkward really quickly, forcing us to find engineering tricks for fusing/concatenating the dual input.

### Precision-weighted merging to the rescue

It turns out that you don't need to compute \\(q\_\phi(z\_i \mid z\_{i+1}, x)\\) directly. Here, we take advantage of the fact that the true posterior \\(p\_\theta(z\_i \mid z\_{i+1}, x)\\) is proportional to its prior and likelihood

$$
\begin{align}
p_\theta(z_i \mid z_{i+1}, x) \propto p_\theta(z_i \mid z_{i+1}) p_\theta(x \mid z_i).
\end{align}
$$

For fixed \\(x\\), we can think of \\(p(x \mid z_i)\\) as \\(\ell(z_i ; x)\\). Since we already have \\(p\_\theta(z\_i \mid z\_{i+1})\\), it makes sense to approximate \\(\ell(z_i ; x)\\) directly!

However, there are two big challenges to the proposed approach. First, our neural network(s) must take \\(x\\) as input and propose *an entire distribution* over \\(z_i\\). Second, we will have to deal with the normalization term somehow (the proportionality sign rears its ugly head). The ladder VAE paper addresses both issues simultaneously by imposing a (strong) assumption: \\(\ell(z_i ; x)\\), when normalized, can be approximated with a Gaussian distribution \\(q_\phi(z_i \mid x)\\).


