<?php
$left = json_decode(file_get_contents('layout-left.json'));
$right = json_decode(file_get_contents('layout-right.json'));

$leftPoints  = array_column($left, 'point');
$rightPoints = array_column($right, 'point');

$leftMinX = min(array_column($leftPoints, 0));
$leftMaxX = max(array_column($leftPoints, 0));

$rightMinX = min(array_column($rightPoints, 0));
$rightMaxX = max(array_column($rightPoints, 0));

//printf("Left %f - %f, Right %f - %f\n", $leftMinX, $leftMaxX, $rightMinX, $rightMaxX);

$output = array();
foreach ($leftPoints as $point) {
  $point[2] = -1.0; // set left Z-coordinate to -1
  $output[] = array('point' => $point);
}
// Flip right points and align them with left points in X
foreach ($rightPoints as $point) {
  $point[2] = 1.0; // set right Z-coordinate to 1
  $point[0] = ($rightMaxX - $point[0]) + $leftMinX;
  $output[] = array('point' => $point);
}

echo json_encode($output);
