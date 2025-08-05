import os
def index_to_position(point_index,y_bias=0,z_bias=0.2):
    if point_index>0:
        row=point_index//10
        col=point_index%10
        y=(col)*0.1+y_bias+0.05
        z=(row)*0.1+z_bias+0.05
        return [0,y,z]
    else:
        point_index=-point_index
        row = point_index // 10
        col = point_index % 10
        # print(col,row)
        y = -(col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05
        return [0, y, z]

def position_to_index(position, y_bias=0, z_bias=0.2):
    """Convert a 3D position to a grid index."""
    y = position[1] - y_bias - 0.05
    z = position[2] - z_bias - 0.05
    row = int(z // 0.1)
    col = int(y // 0.1)
    print("row:", row, "col:", col, "y:", y, "z:", z)
    if y>=0:
        return -row * 10 + col
    else:
        return row * 10 - col

model_file_list = ["/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_mode_c.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_mode_l.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_mode_d.csv"]
RGB_event_file_list = ["/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_c.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_l.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_d.csv"]
event_file_list = ["/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_event_c.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_event_l.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_event_d.csv"]
RGB_filr_list=["/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_c.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_l.csv","/home/xinchi/Event_RGB_catching/AMTTrack/impact_point_RGB_d.csv"]
label_file_list = ["/home/xinchi/Event_RGB_catching/AMTTrack/label_c.csv","/home/xinchi/Event_RGB_catching/AMTTrack/label_l.csv","/home/xinchi/Event_RGB_catching/AMTTrack/label_d.csv"]
predict_file=event_file_list[1]
label_file=label_file_list[1]
with open(predict_file, 'r') as f_predict,open(label_file, 'r') as f_label:
    predict_lines = f_predict.readlines()
    label_lines = f_label.readlines()

    total_count = 0
    correct_count = 0

    predict_i=0
    last_exp_i = 1
    last_predict_position = [0, 0, 0]
    while predict_i < len(predict_lines):
        predict_line = predict_lines[predict_i].strip()
        predict_data = predict_line.split(',')
        exp_i= int(predict_data[0])
        if exp_i==last_exp_i:
            last_predict_position = [float(predict_data[i]) for i in range(2,5)]
            predict_i+=1
            continue
        else:
            label_grid = int(label_lines[last_exp_i-1].strip())
            last_exp_i = exp_i
            if label_grid == 0:
                continue
            print("last predict position",last_predict_position)
            predict_grid=position_to_index(last_predict_position)
            print("predict grid:", predict_grid)
            if predict_grid == label_grid:
                correct_count += 1
            total_count += 1
            print("predict data:", predict_data)
            last_predict_position = [float(predict_data[i]) for i in  range(2,5)]
            print(last_predict_position)
            predict_i += 1
        print("predict position:", last_predict_position)
    print("Accuracy: {:.2f}%".format((correct_count / total_count) * 100 if total_count > 0 else 0))



