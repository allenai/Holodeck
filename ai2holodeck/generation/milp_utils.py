import cvxpy as cp


M = 1e6  # A large number, should be chosen carefully depending on the context
EPSILON = 1e-2


def create_boundary_constraints(c, object_dim, bbox):
    room_min_x, room_min_y, room_max_x, room_max_y = bbox
    x_size, y_size = object_dim
    # Decision variables for object centers
    cx, cy, rotate_90 = c[0], c[1], c[3]

    # Half-sizes considering rotation
    half_width = cp.multiply(0.5 * x_size, 1 - rotate_90) + cp.multiply(
        0.5 * y_size, rotate_90
    )
    half_height = cp.multiply(0.5 * y_size, 1 - rotate_90) + cp.multiply(
        0.5 * x_size, rotate_90
    )

    # Constraints
    constraints = [
        # Constraints for binary variable activation
        cx - half_width >= room_min_x,
        cx + half_width <= room_max_x,
        cy - half_height >= room_min_y,
        cy + half_height <= room_max_y,
    ]
    return constraints


def create_directional_constraints(c1, c2, object_dim_1, object_dim_2):
    x_size1, y_size1 = object_dim_1
    x_size2, y_size2 = object_dim_2
    # Decision variables for object centers
    cx1 = c1[0]
    cy1 = c1[1]
    rotate_180_1 = c1[2]
    rotate_90_1 = c1[3]

    cx2 = c2[0]
    cy2 = c2[1]
    rotate_180_2 = c2[2]
    rotate_90_2 = c2[3]

    # Half-sizes considering rotation
    half_width1 = cp.multiply(0.5 * x_size1, 1 - rotate_90_1) + cp.multiply(
        0.5 * y_size1, rotate_90_1
    )
    half_height1 = cp.multiply(0.5 * y_size1, 1 - rotate_90_1) + cp.multiply(
        0.5 * x_size1, rotate_90_1
    )

    half_width2 = cp.multiply(0.5 * x_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * y_size2, rotate_90_2
    )
    half_height2 = cp.multiply(0.5 * y_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * x_size2, rotate_90_2
    )

    # Binary variables to determine the relative positions
    left_of = cp.Variable(boolean=True)
    right_of = cp.Variable(boolean=True)
    above = cp.Variable(boolean=True)
    below = cp.Variable(boolean=True)

    # Constraints
    constraints = [
        # Constraints for binary variable activation
        # object 1 is left of object 2
        cx2 - cx1 >= EPSILON + half_width1 + half_width2 - M * (1 - left_of),
        # object 1 is right of object 2
        cx1 - cx2 >= EPSILON + half_width1 + half_width2 - M * (1 - right_of),
        # object 1 is below object 2
        cy2 - cy1 >= EPSILON + half_height1 + half_height2 - M * (1 - below),
        # object 1 is above object 2
        cy1 - cy2 >= EPSILON + half_height1 + half_height2 - M * (1 - above),
        # Ensure that at least one of the binary variables must be True
        left_of + right_of + above + below >= 1,
        # make sure the object does not face the wall
        # when "left of" is true, the object should not face left
        rotate_90_1 + rotate_180_1 <= 1 + M * (1 - left_of),
        # when "right of" is true, the object should not face right
        rotate_90_1 + (1 - rotate_180_1) <= 1 + M * (1 - right_of),
        # when "below" is true, the object should not be looking down
        (1 - rotate_90_1) + rotate_180_1 <= 1 + M * (1 - below),
        # when "above" is true, you should not be looking up
        (1 - rotate_90_1) + (1 - rotate_180_1) <= 1 + M * (1 - above),
    ]
    return constraints


def create_nooverlap_constraints(c1, c2, object_dim_1, object_dim_2):
    x_size1, y_size1 = object_dim_1
    x_size2, y_size2 = object_dim_2
    # Decision variables for object centers
    cx1 = c1[0]
    cy1 = c1[1]
    # rotate_180_1 = c1[2]
    rotate_90_1 = c1[3]

    cx2 = c2[0]
    cy2 = c2[1]
    # rotate_180_2 = c2[2]
    rotate_90_2 = c2[3]

    # Half-sizes considering rotation
    half_width1 = cp.multiply(0.5 * x_size1, 1 - rotate_90_1) + cp.multiply(
        0.5 * y_size1, rotate_90_1
    )
    half_height1 = cp.multiply(0.5 * y_size1, 1 - rotate_90_1) + cp.multiply(
        0.5 * x_size1, rotate_90_1
    )

    half_width2 = cp.multiply(0.5 * x_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * y_size2, rotate_90_2
    )
    half_height2 = cp.multiply(0.5 * y_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * x_size2, rotate_90_2
    )

    # Binary variables to determine the relative positions
    left_of = cp.Variable(boolean=True)
    right_of = cp.Variable(boolean=True)
    above = cp.Variable(boolean=True)
    below = cp.Variable(boolean=True)

    # Constraints
    constraints = [
        # Constraints for binary variable activation
        # object 1 is left of object 2
        cx2 - cx1 >= EPSILON + half_width1 + half_width2 - M * (1 - left_of),
        # object 1 is right of object 2
        cx1 - cx2 >= EPSILON + half_width1 + half_width2 - M * (1 - right_of),
        # object 1 is below object 2
        cy2 - cy1 >= EPSILON + half_height1 + half_height2 - M * (1 - below),
        # object 1 is above object 2
        cy1 - cy2 >= EPSILON + half_height1 + half_height2 - M * (1 - above),
        # Ensure that at least one of the binary variables must be True
        left_of + right_of + above + below >= 1,
    ]
    return constraints


def create_alignment_constraints(c1, c2, object_dim_1, object_dim_2):
    x_size1, y_size1 = object_dim_1
    x_size2, y_size2 = object_dim_2
    # Decision variables for object centers
    cx1 = c1[0]
    cy1 = c1[1]
    # rotate_180_1 = c1[2]
    rotate_90_1 = c1[3]

    cx2 = c2[0]
    cy2 = c2[1]
    # rotate_180_2 = c2[2]
    rotate_90_2 = c2[3]

    # Half-sizes considering rotation
    # Binary variables to determine the relative positions
    x_aligned = cp.Variable(boolean=True)
    y_aligned = cp.Variable(boolean=True)

    # Constraints
    constraints = [
        # Constraints for binary variable activation
        cx2 - cx1 <= M * (1 - x_aligned),
        cx1 - cx2 <= M * (1 - x_aligned),
        cy2 - cy1 <= M * (1 - y_aligned),
        cy1 - cy2 <= M * (1 - y_aligned),
        # Ensure that at least one of the binary variables must be True
        x_aligned + y_aligned >= 1,
    ]
    return constraints


def create_edge_constraints(var, object_dim, room_dim, hard=True, use_longer_edge=True):
    x = var[0]
    y = var[1]
    rotate_180 = var[2]
    rotate_90 = var[3]
    x_size, y_size = object_dim
    room_min_x, room_min_y, room_max_x, room_max_y = room_dim

    # Half-sizes considering rotation
    half_width1 = cp.multiply(0.5 * x_size, 1 - rotate_90) + cp.multiply(
        0.5 * y_size, rotate_90
    )
    half_height1 = cp.multiply(0.5 * y_size, 1 - rotate_90) + cp.multiply(
        0.5 * x_size, rotate_90
    )

    a = room_min_x + half_width1
    b = room_max_x - half_width1
    c = room_min_y + half_height1
    d = room_max_y - half_height1

    # x == a or x == b or y ==c or y == d
    # Binary variables for each condition
    bx_a = cp.Variable(boolean=True)
    bx_b = cp.Variable(boolean=True)
    by_c = cp.Variable(boolean=True)
    by_d = cp.Variable(boolean=True)

    # Constraints that link the binary variables with the conditions
    # Constraints
    if hard:
        hard_constraints = [
            # If x_size_longer is true (1), then half_width1 must be greater than half_height1
            # half_width1 - half_height1 >= -M * (1 - x_size_longer),
            # If x_size_longer is false (0), then half_width1 must not be greater than half_height1
            # half_width1 - half_height1 <= M * x_size_longer,
            # Constraints for binary variable activation
            x - a <= M * (1 - bx_a),
            a - x <= M * (1 - bx_a),
            x - b <= M * (1 - bx_b),
            b - x <= M * (1 - bx_b),
            y - c <= M * (1 - by_c),
            c - y <= M * (1 - by_c),
            y - d <= M * (1 - by_d),
            d - y <= M * (1 - by_d),
            # make sure the object does not face the wall
            # 0: facing up, 90: facing right ...
            # when bx_a is true, the left edge is aligned with the left wall
            rotate_90 + rotate_180 <= 1 + M * (1 - bx_a),
            # when bx_b is true, the right edge is aligned with the right wall
            # when bx_b is true, rotate_90 = 1 and rotate_180 = 0 is invalid
            rotate_90 + (1 - rotate_180) <= 1 + M * (1 - bx_b),
            # when by_c is true, the bottom edge is aligned with the bottom wall
            # when by_c is true, rotate_90 = 0 and rotate_180 = 1 is invalid
            (1 - rotate_90) + rotate_180 <= 1 + M * (1 - by_c),
            # when by_d is true, the top edge is aligned with the top wall
            (1 - rotate_90) + (1 - rotate_180) <= 1 + M * (1 - by_d),
            # Only one of the binary variables needs to be true (logical OR)
            bx_a + bx_b + by_c + by_d >= 1,
        ]

        if use_longer_edge:
            # x_size longer should be true if and only if half_width1 > half_height1
            x_size_longer = cp.Variable(boolean=True)
            hard_constraints += [
                # If x_size_longer is true (1), then half_width1 must be greater than half_height1
                half_width1 - half_height1 >= -M * (1 - x_size_longer),
                # If x_size_longer is false (0), then half_width1 must not be greater than half_height1
                half_width1 - half_height1 <= M * x_size_longer,
                by_c + by_d >= 1 - M * (1 - x_size_longer),
                bx_a + bx_b >= 1 - M * x_size_longer,
            ]

        return hard_constraints, [bx_a + bx_b + by_c + by_d]
    else:
        hard_constraints = [
            # Constraints for binary variable activation
            x - a <= M * (1 - bx_a),
            a - x <= M * (1 - bx_a),
            x - b <= M * (1 - bx_b),
            b - x <= M * (1 - bx_b),
            y - c <= M * (1 - by_c),
            c - y <= M * (1 - by_c),
            y - d <= M * (1 - by_d),
            d - y <= M * (1 - by_d),
            # when bx_a is true, the left edge is aligned with the left wall
            rotate_90 + rotate_180 <= 1 + M * (1 - bx_a),
            # when bx_b is true, the right edge is aligned with the right wall
            # when bx_b is true, rotate_90 = 1 and rotate_180 = 0 is invalid
            rotate_90 + (1 - rotate_180) <= 1 + M * (1 - bx_b),
            # when by_c is true, the bottom edge is aligned with the bottom wall
            # when by_c is true, rotate_90 = 0 and rotate_180 = 1 is invalid
            (1 - rotate_90) + rotate_180 <= 1 + M * (1 - by_c),
            # when by_d is true, the top edge is aligned with the top wall
            (1 - rotate_90) + (1 - rotate_180) <= 1 + M * (1 - by_d),
        ]
        return hard_constraints, [bx_a + bx_b + by_c + by_d]


def create_abs_constraints(X, Y, a, constraint_type="geq"):
    """
    Create a constraint for |X - Y| <= a or |X - Y| >= a.

    :param X: The first cvxpy Variable.
    :param Y: The second cvxpy Variable.
    :param a: The non-negative constant a.
    :param constraint_type: A string 'leq' for |X-Y| <= a or 'geq' for |X-Y| >= a.
    :return: A list of one or two cvxpy constraints.
    """
    constraints = []

    if constraint_type == "leq":
        # For |X - Y| <= a, we need two inequalities:
        constraints.append(X - Y <= a)
        constraints.append(Y - X <= a)

    elif constraint_type == "geq":
        # For |X - Y| >= a, we introduce an auxiliary boolean variable to handle the OR condition
        z = cp.Variable(boolean=True)
        # Now we create two constraints that together represent the OR condition
        # If z is True (1), then the first constraint (X - Y >= a) must be satisfied.
        # If z is False (0), then the second constraint (Y - X >= a) must be satisfied.
        constraints.append((X - Y) - M * z >= a - M)
        constraints.append((Y - X) - M * (1 - z) >= a - M)

    else:
        raise ValueError("Invalid constraint_type. Use 'leq' or 'geq'.")

    return constraints


def create_distance_constraints(c1, c2, upper_bound, type="near"):
    X1, Y1 = c1[0], c1[1]
    X2, Y2 = c2[0], c2[1]
    if type == "near":
        # Auxiliary variables for the absolute differences
        abs_diff_x = cp.Variable()
        abs_diff_y = cp.Variable()

        # Constraints for the absolute values
        hard_constraints = [
            abs_diff_x >= X1 - X2,
            abs_diff_x >= X2 - X1,
            abs_diff_y >= Y1 - Y2,
            abs_diff_y >= Y2 - Y1,
            abs_diff_x >= 0,
            abs_diff_y >= 0,
            abs_diff_x <= upper_bound[0],
            abs_diff_y <= upper_bound[1],
        ]

        # L1 distance is the sum of the absolute differences
        l1_distance = abs_diff_x + abs_diff_y
        soft_constraints = [-l1_distance]

    elif type == "far":
        x_lower_bound = cp.Variable()
        y_lower_bound = cp.Variable()
        # Maximize L1 distance
        X1_larger = cp.Variable(boolean=True)
        Y1_larger = cp.Variable(boolean=True)
        # Constraints for the absolute values
        hard_constraints = [
            # if X1 is larger, then X1 - X2 >= x_lower_bound
            X1 - X2 >= x_lower_bound - M * (1 - X1_larger),
            X2 - X1 >= x_lower_bound - M * X1_larger,
            Y1 - Y2 >= y_lower_bound - M * (1 - Y1_larger),
            Y2 - Y1 >= y_lower_bound - M * Y1_larger,
            x_lower_bound >= 0,
            y_lower_bound >= 0,
        ]
        soft_constraints = [x_lower_bound + y_lower_bound]
    else:
        raise ValueError("type must be 'near' or 'far'")

    # Return the objective and the constraints together
    return hard_constraints, soft_constraints


def create_if_and_constraints(A, B):
    # A and B are binary conditions
    #  A and B are true if and only if z is true
    z = cp.Variable(boolean=True)  # New binary variable for the AND condition
    constraints = [z <= A, z <= B, z >= A + B - 1]
    return constraints, z


def create_relative_constraints(c1, c2, object_dim_1, object_dim_2, constraint_type):
    x_size1, y_size1 = object_dim_1
    x_size2, y_size2 = object_dim_2

    # Decision variables for object centers
    cx1 = c1[0]
    cy1 = c1[1]
    # rotate_180_1 = c1[2]
    # rotate_90_1 = c1[3]

    # target object
    cx2 = c2[0]
    cy2 = c2[1]
    rotate_180_2 = c2[2]
    rotate_90_2 = c2[3]

    # Half-sizes considering rotation
    # half_xwidth1 = cp.multiply(0.5 * x_size1, 1 - rotate_90_1) + cp.multiply(0.5 * y_size1, rotate_90_1)
    # half_yheight1 = cp.multiply(0.5 * y_size1, 1 - rotate_90_1) + cp.multiply(0.5 * x_size1, rotate_90_1)

    half_xwidth2 = cp.multiply(0.5 * x_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * y_size2, rotate_90_2
    )
    half_yheight2 = cp.multiply(0.5 * y_size2, 1 - rotate_90_2) + cp.multiply(
        0.5 * x_size2, rotate_90_2
    )

    hard_constraints = []
    soft_constraints = []

    if constraint_type == "left of":
        # if rotate_90_2 == 0 and rotate_180_2 == 0, face up
        constraints, z = create_if_and_constraints(1 - rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        # constraints activated by z being true
        hard_constraints.extend(
            [
                cx1 <= cx2 + M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        ## if rotate_90_2 == 1 and rotate_180_2 == 0, face right
        constraints, z = create_if_and_constraints(rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 >= cy2 - M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 0 and rotate_180_2 == 1, face down
        constraints, z = create_if_and_constraints(1 - rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 >= cx2 - M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 1 and rotate_180_2 == 1, face left
        constraints, z = create_if_and_constraints(rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 <= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )

    if constraint_type == "right of":
        # if rotate_90_2 == 0 and rotate_180_2 == 0, face up
        constraints, z = create_if_and_constraints(1 - rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        # constraints activated by z being true
        hard_constraints.extend(
            [
                cx1 >= cx2 - M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        ## if rotate_90_2 == 1 and rotate_180_2 == 0, face right
        constraints, z = create_if_and_constraints(rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 <= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 0 and rotate_180_2 == 1, face down
        constraints, z = create_if_and_constraints(1 - rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 <= cx2 + M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 1 and rotate_180_2 == 1, face left
        constraints, z = create_if_and_constraints(rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 >= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )

    if constraint_type == "side of":
        # if rotate_90_2 == 0 and rotate_180_2 == 0, face up
        constraints, z = create_if_and_constraints(1 - rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        # constraints activated by z being true
        hard_constraints.extend(
            [
                # cx1 >= cx2 - M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        ## if rotate_90_2 == 1 and rotate_180_2 == 0, face right
        constraints, z = create_if_and_constraints(rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                # cy1 <= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 0 and rotate_180_2 == 1, face down
        constraints, z = create_if_and_constraints(1 - rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                # cx1 <= cx2 + M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 1 and rotate_180_2 == 1, face left
        constraints, z = create_if_and_constraints(rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                # cy1 >= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )

    if constraint_type == "in front of":
        # if rotate_90_2 == 0 and rotate_180_2 == 0, face up
        constraints, z = create_if_and_constraints(1 - rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        # constraints activated by z being true
        hard_constraints.extend(
            [
                cy1 >= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        ## if rotate_90_2 == 1 and rotate_180_2 == 0, face right
        constraints, z = create_if_and_constraints(rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 >= cx2 - M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 0 and rotate_180_2 == 1, face down
        constraints, z = create_if_and_constraints(1 - rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 <= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 1 and rotate_180_2 == 1, face left
        constraints, z = create_if_and_constraints(rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 <= cx2 + M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )

    if constraint_type == "behind":
        # if rotate_90_2 == 0 and rotate_180_2 == 0, face up
        constraints, z = create_if_and_constraints(1 - rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        # constraints activated by z being true
        hard_constraints.extend(
            [
                cy1 <= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        ## if rotate_90_2 == 1 and rotate_180_2 == 0, face right
        constraints, z = create_if_and_constraints(rotate_90_2, 1 - rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 <= cx2 + M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 0 and rotate_180_2 == 1, face down
        constraints, z = create_if_and_constraints(1 - rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cy1 >= cy2 + M * (1 - z),
                cx1 <= cx2 + half_xwidth2 + M * (1 - z),
                cx1 >= cx2 - half_xwidth2 - M * (1 - z),
            ]
        )
        # if rotate_90_2 == 1 and rotate_180_2 == 1, face left
        constraints, z = create_if_and_constraints(rotate_90_2, rotate_180_2)
        hard_constraints.extend(constraints)
        hard_constraints.extend(
            [
                cx1 >= cx2 - M * (1 - z),
                cy1 <= cy2 + half_yheight2 + M * (1 - z),
                cy1 >= cy2 - half_yheight2 - M * (1 - z),
            ]
        )

    return hard_constraints
