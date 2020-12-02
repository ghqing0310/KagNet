import json


def convert_to_statements(filename):
    # 把一条样本写为另一种格式，标注出一句句子中的target entity 和 source entity，用于主题图的创建
    with open('../dataset/diffwords_' + filename + '.txt', 'r', encoding='utf_8') as file:
        statements = []
        for line in file.readlines():
            statement = {}
            newline = eval(line.strip('\n'))
            s1, s2 = newline['original']
            diff_s1, diff_s2 = newline['diffwords']
            statement['sent'] = s1
            ans = ''
            for ele in diff_s1:
                ans += ' ' + ele
            statement['ans'] = ans
            statement['target'] = diff_s1
            statement['source'] = newline['samewords']
            statements.append(statement)
            statement = {}
            statement['sent'] = s2
            ans = ''
            for ele in diff_s2:
                ans += ' ' + ele
            statement['ans'] = ans
            statement['target'] = diff_s2
            statement['source'] = newline['samewords']
            statements.append(statement)
    with open('../dataset/diffwords_' + filename + '.mcp', 'w') as f:
        json.dump(statements, f)


def convert_to_statements2(filename):
    """
    将一条样本写为：
        {‘statements’: [{'label':True, 'statement':s1}, {'label':False, 'statement':s2}], 'id': xxx}
    """
    with open("../dataset/diffwords_" + filename + ".txt", 'r', encoding='utf_8') as file:
        f = file.readlines()
    with open("../dataset/" + filename + ".statements", "w", encoding='utf_8') as file:
        for line in f:
            newline = eval(line.strip('\n'))
            s = {}
            correct = int(newline["correct"])
            if correct == 0:
                lst = [{"label": True, "statement": newline['original'][0]},
                    {"label": False, "statement": newline['original'][1]}]
            else:
                lst = [{"label": False, "statement": newline['original'][0]},
                    {"label": True, "statement": newline['original'][1]}]
            s["statements"] = lst
            s["id"] = newline['idx']
            file.write(json.dumps(s))
            file.write("\n")


if __name__ == '__main__':

    convert_to_statements('train')
   # convert_to_statements('dev')
    convert_to_statements('test')

    convert_to_statements2('train')
    # convert_to_statements2('dev')
    convert_to_statements2('test')
