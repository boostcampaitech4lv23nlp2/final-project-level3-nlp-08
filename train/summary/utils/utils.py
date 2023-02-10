
sub_label_list = ['주거와 생활', '여행', '교통', '회사/아르바이트', '군대', '교육',
              '가족', '연애/결혼', '반려동물', '스포츠/레저', '게임', '식음료',
              '계절/날씨', '사회이슈', '타 국가 이슈', '미용', '건강', '상거래 전반',
              '방송/연예', '영화/만화']



def sub_label_to_num(label):
    return sub_label_list.index(label)

def sub_num_to_labels(nums):
    return [sub_label_list[num] for num in nums]