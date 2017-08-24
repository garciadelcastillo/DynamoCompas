from __future__ import print_function

import System


class MatlabClient(object):
    """"""

    def __init__(self):
        self.type = None
        self.app = None
        self.lease = None
        self.verbose = False
        self.wsname = 'base'
        self.init()

    def init(self):
        self._create_instance()
        self._init_lease()

    def _create_instance(self):
        self.type = System.Type.GetTypeFromProgID('Matlab.Application')
        self.app = System.Activator.CreateInstance(self.type)
        self.app.Visible = False

    def _init_lease(self):
        self.lease = self.app.InitializeLifetimeService()
        self.lease.InitialLeaseTime = System.TimeSpan.FromMinutes(5.0)
        self.lease.RenewOnCallTime = System.TimeSpan.FromMinutes(5.0)

    def _renew_lease(self):
        self.lease.Renew(System.TimeSpan.FromMinutes(5.0))

    @staticmethod
    def vector_from_list(a, dtype=float):
        n = len(a)
        vector = System.Array.CreateInstance(dtype, n)
        for i in range(n):
            vector[i] = a[i]
        return vector

    @staticmethod
    def matrix_from_list(A, dtype=float):
        m = len(A)
        n = len(A[0])
        if not all([len(row) == n for row in A]):
            raise Exception('Matrix dimensions inconsistent.')
        matrix = System.Array.CreateInstance(dtype, m, n)
        for row in range(m):
            for col in range(n):
                matrix[row, col] = A[row][col]
        return matrix

    @staticmethod
    def list_from_vector(a):
        return list(a)

    @staticmethod
    def list_from_matrix(A, m, n):
        nlist = []
        for row in range(m):
            nlist.append([None] * n)
            for col in range(n):
                nlist[row][col] = A[row, col]
        return nlist

    def eval(self, cmd):
        res = self.app.Execute(cmd)
        if self.verbose:
            print(res)
        self._renew_lease()

    def put(self, name, value):
        try:
            res = self.app.PutFullMatrix(name, self.wsname, value, None)
        except Exception:
            res = self.app.PutWorkspaceData(name, self.wsname, value)
        if self.verbose:
            print(res)
        self._renew_lease()

    def get(self, name):
        _value = self.app.GetVariable(name, self.wsname)
        try:
            _value.Rank
        except AttributeError:
            value = _value
        else:
            value = []
            if _value.Rank == 1:
                value = MatlabClient.list_from_vector(_value)
            elif _value.Rank == 2:
                m, n = self.get_matrix_size(name)
                value = MatlabClient.list_from_matrix(_value, m, n)
        self._renew_lease()
        return value

    def get_vector(self, name):
        _value = self.app.GetVariable(name, self.wsname)
        try:
            _value.Rank
        except AttributeError:
            value = _value
        else:
            value = MatlabClient.list_from_vector(_value)
        self._renew_lease()
        return value

    def get_matrix_size(self, name):
        self.app.Execute('[m, n] = size({0});'.format(name))
        m = self.app.GetVariable('m', self.wsname)
        n = self.app.GetVariable('n', self.wsname)
        return int(m), int(n)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    matlab = MatlabClient()

    A = matlab.matrix_from_list([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])

    matlab.put('A', A)
    matlab.eval('[R, jb] = rref(A);')

    R = matlab.get('R')
    jb = matlab.get('jb')

    print(R)
    print(jb)
