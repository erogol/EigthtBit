import inspect

class Pipeline():
	"""
		Construct processing pipeline. It is useful for image preprocessing.
	"""
	def __init__(self,ops = [], arg_list = []):
		"""
			ops 		:	a tuple list (operation name, operation with transform function)
			arg_list 	: 	a list of dictionaries. Each dictionary includes keyworded
							function arguments of the corresponding function at ops list
		"""
		self.ops = ops
		self.arg_list = arg_list

	def transform(self, data, verbose = False):
		aux_return_vals = []
		if len(self.arg_list) > 0:
			count = 0
			for op_name, op in self.ops:
				if verbose:
					print op_name
				if 'self' in inspect.getargspec(op)[0]:
					data_arg = {inspect.getargspec(op)[0][1]:data}
				else:
					data_arg = {inspect.getargspec(op)[0][0]:data}
				#print data_arg
				args = dict(data_arg.items() + self.arg_list[count].items())
				#print args
				data = op(**args)
				if type(data) is list or type(data) is tuple:
					aux_vals = data[1:]
					data = data[0]
					aux_return_vals .append((op_name, aux_vals))
				count += 1
		else:
			for op_name, op in self.ops:
				if verbose:
					print op_name
				data = op(data)
		if len(aux_return_vals) > 0:
			return data, aux_return_vals
		return (data,)