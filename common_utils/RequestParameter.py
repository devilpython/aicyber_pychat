# -*- coding: UTF-8 -*-

#获得参数
def getParameter(request, parameter_name):
    result = None
    try:
        result = request.args.get(parameter_name)
    except:
        pass
    if result is None:
        try:
            if parameter_name in request.json:
                result = request.json[parameter_name]
        except:
            pass
    if result is None and parameter_name == 'token':
        for key in request.headers.keys():
            if key.lower() == 'token':
                result = request.headers.get(key)
            elif key.lower() == 'authorization':
                result = request.headers.get(key)
                token_list = result.split(' ')
                if len(token_list) > 1:
                    result = token_list[1]
                else:
                    result = token_list[0]
    return result