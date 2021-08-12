
def is_inside(polygon, point):

    def cross(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if y1-y2 == 0:
            if y1 == point[1]:
                if min(x1, x2) <= point[0] <= max(x1, x2):
                    return 1, True
            return 0, False

        if x1 - x2 == 0:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                if point[0] <= max(x1, x2):
                    return 1, point[0] == max(x1, x2)
            return 0, False

        a = (y1 - y2) / (x1 - x2)
        b = y1 - x1 * a
        x = (point[1] - b) / a
        if point[0] <= x:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                return 1, point[0] == x or point[1] in (y1,y2)
        return 0, False

    cross_points = 0
    for x in range(len(polygon)):
        num, on_line = cross(polygon[x], polygon[x-1])
        if on_line:
            return True
        cross_points += num
    return cross_points % 2
polygon = [[1275, 335], [506, 211], [250, 212], [254, 573], [1275, 586]]
point = [300,586]
print(is_inside(polygon,point))