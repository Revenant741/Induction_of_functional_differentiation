import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from functools import reduce


class HessianFree(torch.optim.Optimizer):
    """
    `Training Deepandで提示されたヘッセ行列のないアルゴリズムを実装します
    ヘッセ行列のない最適化を使用したリカレントネットワーク `_。
    
    引数：
    params（iterable）：最適化するパラメーターの反復可能または定義の指示
        パラメータグループ
    lr（フロート、オプション）：学習率（デフォルト：1）
    delta_decay（float、オプション）：
        の前の結果の減衰の共役勾配法によるデルタの計算
        次の共役勾配反復の初期化
    damping（フロート、オプション）：
        Tikhonovダンピングの初期値
        係数。（デフォルト：0.5）
    max_iter（int、オプション）：共役勾配の最大数
        反復（デフォルト：50）
    use_gnm（bool、オプション）：一般化されたガウス-ニュートン行列を使用します。
        おそらくヘッセ行列の不確定性を解決します（セクション20.6）
    verbose（bool、オプション）：ステートメントの出力（デバッグ）
.. _ヘッセ行列のない最適化を使用したディープリカレントネットワークのトレーニング：
    https://doi.org/10.1007/978-3-642-35289-8_27
    """

    def __init__(self, params,
                 lr=1,
                 damping=0.5,
                 delta_decay=0.95,
                 cg_max_iter=100,
                 use_gnm=True,
                 verbose=False):

        if not (0.0 < lr <= 1):
            raise ValueError("Invalid lr: {}".format(lr))

        if not (0.0 < damping <= 1):
            raise ValueError("Invalid damping: {}".format(damping))

        if not cg_max_iter > 0:
            raise ValueError("Invalid cg_max_iter: {}".format(cg_max_iter))

        defaults = dict(alpha=lr,
                        damping=damping,
                        delta_decay=delta_decay,
                        cg_max_iter=cg_max_iter,
                        use_gnm=use_gnm,
                        verbose=verbose)
        super(HessianFree, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HessianFree doesn't support per-parameter options (parameter groups)")
                #HessianFree はパラメータごとのオプション (パラメータグループ) をサポートしていません

        self._params = self.param_groups[0]['params']

    def _gather_flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def step(self, closure, b=None, M_inv=None):
        '''
        単一の最適化ステップを実行します。
        引数：
            closure（呼び出し可能）：モデルを再評価するクロージャ
                損失と出力のタプルを返します。
            b（呼び出し可能、オプション）：ベクトルbを計算するクロージャ
                最小化問題x ^ T.A.x + x^Tb。
            M（呼び出し可能、オプション）：AのINVERSE前提条件
        '''
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        cg_max_iter = group['cg_max_iter']
        damping = group['damping']
        use_gnm = group['use_gnm']
        verbose = group['verbose']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        loss_before, output = closure()
        current_evals = 1
        state['func_evals'] += 1

        # 現在のパラメータとそれぞれのグラデーションを収集
        flat_params = parameters_to_vector(self._params)
        flat_grad = self._gather_flat_grad()

        # 線形演算子の定義
        if use_gnm:
            # 一般化ガウスニュートンベクトル積
            def A(x):
                return self._Gv(loss_before, output, x, damping)
        else:
            # ヘシアンベクトル積
            def A(x):
                return self._Hv(flat_grad, x, damping)

        if M_inv is not None:
            m_inv = M_inv()

            # プリコンディショナーのレシピ (論文13章)
            if m_inv.dim() == 1:
                m = (m_inv + damping) ** (-0.85)

                def M(x):
                    return m * x
            else:
                m = torch.inverse(m_inv + damping * torch.eye(*m_inv.shape))

                def M(x):
                    return m @ x
        else:
            M = None

        b = flat_grad.detach() if b is None else b().detach().flatten()

        # 共役勾配の初期化 (論文10)
        if state.get('init_delta') is not None:
            init_delta = delta_decay * state.get('init_delta')
        else:
            init_delta = torch.zeros_like(flat_params)

        eps = torch.finfo(b.dtype).eps

        # 共役勾配
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,
                              M=M, max_iter=cg_max_iter,
                              tol=1e1 * eps, eps=eps, martens=True)

        # パラメータの更新
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]

        vector_to_parameters(flat_params + delta, self._params)
        loss_now = closure()[0]
        current_evals += 1
        state['func_evals'] += 1

        # 共役勾配バックトラッキング (20.8.7項)
        if verbose:
            #print("Loss before CG: {}".format(float(loss_before)))
            #print("Loss before BT: {}".format(float(loss_now)))
            pass
        for (d, m) in zip(reversed(deltas[:-1][::2]), reversed(Ms[:-1][::2])):
            vector_to_parameters(flat_params + d, self._params)
            loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            M = m
            loss_now = loss_prev

        if verbose:
            #print("Loss after BT:  {}".format(float(loss_now)))
            pass
        # レーベンバーグマーカートヒューリスティック (Section 20.8.5)
        reduction_ratio = (float(loss_now) -
                           float(loss_before)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3
        if reduction_ratio < 0:
            group['init_delta'] = 0

        # ライン探索 (Section 20.8.8)
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)

        for _ in range(60):
            if float(loss_now) <= float(loss_before) + alpha * min_improv:
                break

            alpha *= beta
            vector_to_parameters(flat_params + alpha * delta, self._params)
            loss_now = closure()[0]
        else:  # No good update found
            alpha = 0.0
            loss_now = loss_before

        # Update the parameters (this time fo real)
        vector_to_parameters(flat_params + alpha * delta, self._params)

        if verbose:
            '''
            print("Loss after LS:  {0} (lr: {1:.3f})".format(
                float(loss_now), alpha))
            print("Tikhonov damping: {0:.3f} (reduction ratio: {1:.3f})".format(
                group['damping'], reduction_ratio), end='\n\n')
            '''
        return loss_now

    def _CG(self, A, b, x0, M=None, max_iter=50, tol=1.2e-6, eps=1.2e-7,
            martens=False):
        """
        共役を用いて線形系 x^T.A.x - x^T b を最小化します。
            勾配法

        引数。
            A (呼び出し可能)。を実装した抽象線形演算子
                積 A.x. A は， hermitian の正定値を表さなければならない．
                マトリックスを使用しています。
            b (torch.Tensor)。ベクトル b。
            x0 (torch.Tensor). x の初期値．
            M (callable, オプション). を実装した抽象線形演算子．
            (A の場合の) 前処理行列とベクトルの積.
            tol (float, オプション). 収束の許容値．
            martens (bool, オプション). Martensの収束基準のフラグ．
        """

        x = [x0]
        r = A(x[0]) - b

        if M is not None:
            y = M(r)
            p = -y
        else:
            p = -r

        res_i_norm = r @ r

        if martens:
            m = [0.5 * (r - b) @ x0]

        for i in range(max_iter):
            Ap = A(p)

            alpha = res_i_norm / ((p @ Ap) + eps)

            x.append(x[i] + alpha * p)
            r = r + alpha * Ap

            if M is not None:
                y = M(r)
                res_ip1_norm = y @ r
            else:
                res_ip1_norm = r @ r

            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm

            # Martens' Relative Progress stopping condition (Section 20.4)
            if martens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])

                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break

            if res_i_norm < tol or torch.isnan(res_i_norm):
                break

            if M is not None:
                p = - y + beta * p
            else:
                p = - r + beta * p

        return (x, m) if martens else (x, None)

    def _Hv(self, gradient, vec, damping):
        """
        Computes the Hessian vector product.
        """
        Hv = self._Rop(gradient, self._params, vec)

        # Tikhonov damping (Section 20.8.1)
        return Hv.detach() + damping * vec

    def _Gv(self, loss, output, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = self._Rop(output, self._params, vec)

        gradient = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._Rop(gradient, output, Jv)

        JHJv = torch.autograd.grad(
            output, self._params, grad_outputs=HJv.reshape_as(output), retain_graph=True)

        # Tikhonov damping (Section 20.8.1)
        return parameters_to_vector(JHJv).detach() + damping * vec

    @staticmethod
    def _Rop(y, x, v, create_graph=False):
        """
        Computes the product (dy_i/dx_j) v_j: R-operator
        """
        if isinstance(y, tuple):
            ws = [torch.zeros_like(y_i, requires_grad=True) for y_i in y]
        else:
            ws = torch.zeros_like(y, requires_grad=True)

        jacobian = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True)

        Jv = torch.autograd.grad(parameters_to_vector(
            jacobian), ws, grad_outputs=v, create_graph=create_graph)

        return parameters_to_vector(Jv)


# The empirical Fisher diagonal (Section 20.11.3)
def empirical_fisher_diagonal(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grads.append(torch.autograd.grad(fi, net.parameters(),
                                         retain_graph=False))

    vec = torch.cat([(torch.stack(p) ** 2).mean(0).detach().flatten()
                     for p in zip(*grads)])
    return vec


# The empirical Fisher matrix (Section 20.11.3)
def empirical_fisher_matrix(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grad = torch.autograd.grad(fi, net.parameters(),
                                   retain_graph=False)
        grads.append(torch.cat([g.detach().flatten() for g in grad]))

    grads = torch.stack(grads)
    n_batch = grads.shape[0]
    return torch.einsum('ij,ik->jk', grads, grads) / n_batch
