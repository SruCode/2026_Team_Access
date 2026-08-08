class HazardZone:
    def __init__(self, x_min=400, y_min=100, x_max=600, y_max=400):
        # Frame Box boundaries (Pixel values)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def is_inside(self, point_x, point_y):
        """Checks whether point (e.g. wrist/hand) is inside danger zone"""
        return (self.x_min <= point_x <= self.x_max) and (self.y_min <= point_y <= self.y_max)