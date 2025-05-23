import React from "react";
import styled from "styled-components";

const StyledSection04 = styled.img`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  margin: 0 auto;
`;

export const Section04 = ({ imageUrl }) => {
  return <StyledSection04 src={imageUrl} alt="Section 04" />;
};
