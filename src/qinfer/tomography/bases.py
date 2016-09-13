#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# bases.py: Representations of Hermitian bases for tomography.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@cgranade.com).
# Based on work with Joshua Combes (joshua.combes@gmail.com).
#     
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

# TODO: docstrings!
# TODO: unit tests!

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

## IMPORTS ###################################################################

from builtins import range, map, str
from functools import reduce

import itertools as it

import numpy as np

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
from qinfer.utils import get_qutip_module
qt = get_qutip_module('3.2')

## EXPORTS ###################################################################

__all__ = [
    'gell_mann_basis',
    'pauli_basis',
    'tensor_product_basis',
    'TomographyBasis'
]

## FUNCTIONS #################################################################

def gell_mann_basis(dim):
    """    
    Returns a :class:`~qinfer.tomography.TomographyBasis` on dim dimensions
    using the generalized Gell-Mann matrices.

    This implementation is based on a MATLAB-language implementation
    provided by Carlos Riofrío, Seth Merkel and Andrew Silberfarb.
    Used with permission.

    :param int dim: Dimension of the individual matrices making up
        the returned basis.
    :rtype: :class:`~qinfer.tomography.TomographyBasis`
    :return: A basis of ``dim * dim`` Gell-Mann matrices.
    """
    # Start by making an empty array of the right shape to
    # hold the matrices that we construct.
    basis = np.zeros((dim**2, dim, dim), dtype=complex)

    # The first matrix should be the identity.
    basis[0, :, :] = np.eye(dim) / np.sqrt(dim)

    # The next dim basis elements should be diagonal,
    # with all by one element nonnegative.
    for idx_basis in range(1, dim):
        basis[idx_basis, :, :] = np.diag(np.concatenate([
            np.ones((idx_basis, )),
            [-idx_basis],
            np.zeros((dim - idx_basis - 1, ))
        ])) / np.sqrt(idx_basis + idx_basis**2)

    # Finally, we get the off-diagonal matrices.
    # These rely on some index gymnastics I don't yet fully
    # understand.
    y_offset = dim * (dim - 1) // 2
    for idx_i in range(1, dim):
        for idx_j in range(idx_i):
            idx_basis = (idx_i - 1) * (idx_i) // 2 + idx_j + dim
            basis[idx_basis, [idx_i, idx_j], [idx_j, idx_i]] = 1 / np.sqrt(2)
            basis[idx_basis + y_offset, [idx_i, idx_j], [idx_j, idx_i]] = [1j / np.sqrt(2), -1j / np.sqrt(2)]

    return TomographyBasis(basis, [dim], r'\gamma', name='gell_mann_basis')

def tensor_product_basis(*bases):
    """
    Returns a TomographyBasis formed by the tensor
    product of two or more factor bases. Each basis element
    is the tensor product of basis elements from the underlying
    factors.
    """
    dim = np.prod([basis.data.shape[1] for basis in bases])
    tp_basis = np.zeros((dim**2, dim, dim), dtype=complex)

    for idx_factors, factors in enumerate(it.product(*[basis.data for basis in bases])):
        tp_basis[idx_factors, :, :] = reduce(np.kron, factors)

    return TomographyBasis(tp_basis,
        sum((
            factor.dims for factor in bases
        ), []),
        list(map(
        r"\otimes".join,
        it.product(*[
            basis.labels for basis in bases
        ])
    )))

def pauli_basis(nq=1):
    """
    Returns a TomographyBasis for the Pauli basis on ``nq``
    qubits.

    :param int nq: Number of qubits on which the returned
        basis is defined.
    """
    basis = tensor_product_basis(*[
        TomographyBasis(
            gell_mann_basis(2).data[[0, 2, 3, 1]],
            [2],
            [u'𝟙', r'\sigma_x', r'\sigma_y', r'\sigma_z']
        )
    ] * nq)

    basis._name = 'pauli_basis'
    return basis

def _format_float_as_latex(c, tol=1e-10):
    if abs(c - int(c)) <= tol:
        return str(int(c))
    elif 1e-3 <= abs(c) <= 1e3:
        return u"{:0.3f}".format(c)
    else:
        return (u"{:0.3e}".format(c)).replace("e", r"\times10^{") + "}"


def _format_complex_as_latex(c, tol=1e-10):
    if abs(c.imag) <= tol:
        # Purely real.
        return _format_float_as_latex(c.real, tol=tol)
    elif abs(c.real) <= tol:
        return _format_float_as_latex(c.imag, tol=tol) + r"\mathrm{i}"
    else:
        return u"{} + {}\mathrm{{i}}".format(
            _format_float_as_latex(c.real, tol=tol),
            _format_float_as_latex(c.imag, tol=tol)
        )


## CLASSES ###################################################################

class TomographyBasis(object):
    """
    A basis of Hermitian operators used for representing tomographic
    objects (states and channels) as vectors of real elements. By assumption,
    a tomographic basis is taken to have an initial (0th) element proportional
    to the identity, and all other elements are taken to be traceless. For
    example, the Pauli matrices form a tomographic basis for qubits.

    Instances of TomographyBasis convert between representations of
    tomographic objects as real vectors of model parameters and QuTiP :class:`~qutip.Qobj`
    instances. The latter is convienent for working with other libraries, and
    for reasoning about fidelities and other metrics, while model parameter
    representations are useful for defining prior distributions and
    tomographic models.

    :param np.ndarray data: Dense array of shape ``(dim ** 2, dim, dim)``
        containing all elements of the new tomographic basis. ``data[alpha, i, j]``
        is the ``(i, j)``-th element of the ``alpha``-th matrix of the new basis.
    :param list dims: Dimensions specification used in converting to QuTiP
        representations. The product of all elements of ``dims`` must equal
        the dimension of axes 1 and 2 of ``data``. For instance, ``[2, 3]``
        specifies that the basis is over the tensor product of a qubit
        and a qutrit space.
    :param labels: LaTeX-formatted labels for each basis element. If a single
        `str`, a subscript is added to each basis element's label.
    :type labels: :obj:`str` or :obj:`list` of :obj:`str`
    :param str superrep: Superoperator representation to pass to QuTiP
        when reconstructing states.
    """
    
    #: Dense matrix... TODO: document indices!
    data = None
    #: Dimensions of each index, used when converting to QuTiP
    #: :class:`~qutip.Qobj` instances.
    dims = None
    #: Labels for each basis element.
    labels = None

    def __init__(self, data, dims, labels=None, superrep=None, name=None):
        self.data = data
        self.dims = dims
        self.superrep = superrep

        dim = self.dim

        self._name = name if name is not None else "(unnamed)"

        if isinstance(labels, str):
            self.labels = list(map("{}_{{}}".format(labels).format, range(dim**2)))
        else:
            self.labels = list(map(r'B_{}'.format, range(dim**2))) if labels is None else labels

        self._flat = self.data.reshape((self.data.shape[0], -1))

    def __repr__(self):
        return "<TomographyBasis {} dims={} at 0x{:0x}>".format(
            self._name, self.dims, id(self)
        )

    def _repr_html_(self):

        if self.dim <= 10:
            element_strings = [r"""
                {label} =                   
                \left(\begin{{matrix}}
                    {rows}
                \end{{matrix}}\right)
                """.format(
                    rows=u"\\\\".join([
                        u"&".join(map(_format_complex_as_latex, row))
                        for row in element
                    ]),
                    label=label
                )
                for element, label in zip(self.data, self.labels)
            ]

            return r"""
            <strong>TomographyBasis:</strong>
                dims=${dims}$
            <p>
                \begin{{equation}}
                    {elements}
                \end{{equation}}
            </p>
            """.format(
                dims=r"\times".join(map(str, self.dims)),
                labels=u",".join(self.labels),
                elements=u",".join(element_strings)
            )
        else:
            return r"""
            <strong>TomographyBasis:</strong>
                dims=${dims}$,
                labels=$\\{{{labels}\\}}$
            """.format(
                dims=r"\times".join(map(str, self.dims)),
                labels=u",".join(self.labels)
            )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return qt.Qobj(self.data[idx], [self.dims, self.dims])
        elif isinstance(idx, list):
            return [self[inner_idx] for inner_idx in idx]
        else:
            raise TypeError("Expected int or list index, not {}.".format(type(idx)))

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return self.dim ** 2

    @property
    def dim(self):
        """
        Dimension of the Hilbert space on which elements of this basis act.

        :type: `int`
        """
        return np.prod(self.dims)

    @property
    def name(self):
        """
        Name to use when converting this basis to a string.

        :type: `str`
        """
        return self._name

    def flat(self):
        r"""
        Returns a NumPy array that represents this operator basis
        in a flattened manner, such that ``basis.flat()[i, j]`` is
        the :math:`j\text{th}` element of the flattened
        :math:`i\text{th}` basis operator.
        """
        return self._flat
    

    def state_to_modelparams(self, state):
        """
        Converts a QuTiP-represented state into a model parameter vector.

        :param qutip.Qobj state: State to be converted.
        :rtype: :class:`np.ndarray`
        :return: The representation of the given state in this basis,
            as a vector of real parameters.
        """
        basis = self.flat()
        data = state.data.todense().view(np.ndarray).flatten()

        # NB: assumes Hermitian state and basis!
        return np.real(np.dot(basis.conj(), data))

    def modelparams_to_state(self, modelparams):
        """
        Converts one or more vectors of model parameters into
        QuTiP-represented states.

        :param np.ndarray modelparams: Array of shape
            ``(basis.dim ** 2, )`` or
            ``(n_states, basis.dim ** 2)`` containing
            states represented as model parameter vectors in this
            basis.
        :rtype: :class:`~qutip.Qobj` or `list` of :class:`~qutip.Qobj`
            instances.
        :return: The given states represented as :class:`~qutip.Qobj`
            instances.
        """
        if modelparams.ndim == 1:
            qobj = qt.Qobj(
                np.tensordot(modelparams, self.data, 1),
                dims=[self.dims, self.dims]
            )
            if self.superrep is not None:
                qobj.superrep = self.superrep
            return qobj
        else:
            return list(map(self.modelparams_to_state, modelparams))

    def covariance_mtx_to_superop(self, mtx):
        """
        Converts a covariance matrix to the corresponding
        superoperator, represented as a QuTiP Qobj
        with ``type="super"``.
        """
        M = self.flat()
        return qt.Qobj(
            np.dot(np.dot(M.conj().T, mtx), M),
            dims=[[self.dims] * 2] * 2
        )
