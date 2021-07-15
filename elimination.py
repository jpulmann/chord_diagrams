import numpy as np
from math import gcd

def eliminate(m, progress=True, step=10):
	print(m.shape)
	rows, columns = m.shape
	column = 0
	row = 0
	
	col_order = np.arange(columns)
	
	if not m.any():
		#~ print("m is empty", m)
		return (m[0, :], col_order)
	if progress:
		print("Column progress: ", end='')
	maxval = 0
	while 1:
		minpos = -1
		for i in range(column, columns):
			if np.any(m[row:, i]):
				if i == column:
					break
				#~ print("Permuting columns %i and %i" % (i, column))
				m[:,[i, column]] = m[:, [column, i]]
				col_order[[i,column]] = col_order[[column, i]]
				break
		else:
			#~ print('All remaining rows are zero')
			break
		for i in range(row, rows):
			if m[i, column] != 0:
				tminval = m[i, column] 
				minpos = i
				break
		else:
			raise ValueError('Shouldn\'t be here! ')
		assert minpos != -1
		m[[minpos, row]] = m[[row, minpos]]
		minval = m[row, column]
		assert minval == tminval
		for i in range(row+1, rows):
			currval = m[i, column]
			if currval != 0:
				gcd_ = gcd(minval, currval)
				alpha = currval//gcd_
				beta = minval//gcd_
				m[i] =  m[i]*beta - m[row]*alpha
		if progress and column % step == 0:
			print('%i ' % (column/step), end='', flush=True)
		row += 1
		column +=1
		
		if row >= rows:
			break
		if column >= columns:
			break
	row -= 1
	column -= 1
	for i in range(row+1):
		assert m[i, i] != 0 , (i)
		assert np.sum(np.abs(m[i+1:, i])) == 0
	for i in range(row+1, rows):
		assert np.sum(np.abs(m[i])) == 0
	#normalize:
	print('Normalize')
	for i in range(row+1):
		diagval = m[i, i]
		for j in range(columns):
			if m[i, j] % m[i, i] != 0:
				#~ print('Line %i not normalizable for j=%i, m[i, i] = %i, m[i, j] = %i' % (i, j, m[i, i], m[i, j]))
				#~ np.savetxt('line.txt', m[i], fmt='%i')
				gcd_ = diagval
				for k in range(columns):
					if m[i,k]:
						gcd_ = gcd(gcd_, m[i, k])
				#~ print('Normalizing at least with %i' %gcd_)
				tmp = m[i]//gcd_
				m[i] = tmp
				break
		else:
			#~ print("normalizing line %i with value %i" % (i, diagval))
			tmp = m[i]//diagval
			for l in range(columns):
				assert m[i, l] % diagval == 0, '%i %i' % (m[i, l], diagval)
			m[i] = tmp
	#~ np.savetxt('matrix.txt', m, fmt='%i')

	#~ print('Now just make it diagonal')
	for i in list(range(row+1))[::-1]:
		if progress and int(100*i/(row+1))*columns//100 == i:
			print('%i%% ' % int(100*i/(row+1)), end='',  flush=True)
		for j in range(i):
			if m[j, i] != 0:
				gcd_ = gcd(m[i,i], m[j,i])
				alpha = m[j, i]//gcd_
				beta = m[i, i]//gcd_
				m[j] = (m[j]*beta - m[i]*alpha)
	if progress:
		print()
	#https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy
	#~ print('Normalize again')
	for i in range(row+1):
		diagval = m[i, i]
		for j in range(row+1, columns):
			if m[i, j] % m[i, i] != 0:
				#~ print('Line %i not normalizable for j=%i, m[i, i] = %i, m[i, j] = %i' % (i, j, m[i, i], m[i, j]))
				#~ np.savetxt('line.txt', m[i], fmt='%i')
				gcd_ = diagval
				for k in range(row+1, columns):
					if m[i,k]:
						gcd_ = gcd(gcd_, m[i, k])
				#~ print('Normalizing at least with %i' %gcd_)
				m[i] = m[i]//gcd_
				break
		else:
			m[i] = m[i]//diagval
	#~ np.savetxt('matrix.txt', m, fmt='%i')
	return (m[~(m==0).all(1).A1], col_order)


if __name__ == "__main__":
	tm = np.matrix('1 1 -1 ; 1 1 -1; 0 1 1; 0 0 0', np.dtype(np.int))
	print(eliminate(tm)[0].shape)
	
