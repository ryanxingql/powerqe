def generate_exp_name(exp_name, params):
    for k, v in params.items():
        if type(v) == int:
            pass
        elif type(v) == list:
            v = [str(iv) for iv in v]
            v = '-'.join(v)
        else:
            raise TypeError
        exp_name += f'_{k}_{v}'
    return exp_name
