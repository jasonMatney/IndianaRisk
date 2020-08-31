## Not run:
require(dynatopmodel)
data("brompton")
# Upslope area and wetness index for Brompton catchment
layers <- build_layers(brompton$dem)
sp::plot(layers, main=c("Elevation AMSL (m)", "Upslope area (log(m^2/m))", "TWI ((log(m^2/m))"))
## End(Not run)