
def cal_position(point_index,y_bias=-0.4,z_bias=0.85):
    point_index = point_index-1
    row=point_index//8
    col=point_index%8
    print(row,col)
    y=(col)*0.1+y_bias+0.05
    z=(8-row)*0.1+z_bias+0.05
    return y,z
print(cal_position(38))