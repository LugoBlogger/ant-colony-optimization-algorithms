import sys

def add_comma(input_file, output_file):

    with open(input_file, 'r') as f:
        data = f.readlines()

    xy_coor = [line.strip('\n') for line in data]
    xy_coor = [line.lstrip().rstrip() for line in xy_coor]
    xy_coor = [line.split(' ') for line in xy_coor]
    xy_coor = [[r.strip() for r in line] for line in xy_coor]
    xy_coor = [', '.join(line) for line in xy_coor]

    with open(output_file, 'w') as f:
        data = '\n'.join(xy_coor)
        f.write(data)


    #print(xy_coor)

    return None


if __name__ == '__main__':
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    add_comma(input_file, output_file)
