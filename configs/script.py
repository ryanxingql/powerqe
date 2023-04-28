def generate_exp_name(exp_name, params):
    for k, v in params.items():
        if isinstance(v, (int, str, bool)):
            pass
        elif type(v) == list:
            v = [str(iv) for iv in v]
            v = '-'.join(v)
        else:
            raise TypeError('Supported types include int/str/bool (list);'
                            f' received `{type(v)}`')
        exp_name += f'_{k}_{v}'
    return exp_name
