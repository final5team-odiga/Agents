import React from "react";
import styled from "styled-components";

const StyledSection09 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  position: relative;
`;

const StyledHeadline = styled.div`
  position: absolute;
  top: 136px;
  left: 53px;
  width: 948px;
  color: black;
  font-size: 50px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

const StyledSubhead = styled.div`
  position: absolute;
  top: 188px;
  left: 53px;
  width: 948px;
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

const StyledParagraph = styled.div`
  position: absolute;
  top: 548px;
  left: 20px;
  width: 1014px;
  text-align: right;
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

export const Section09 = ({ title, subtitle, body }) => {
  return (
    <StyledSection09>
      <StyledHeadline>{title}</StyledHeadline>
      <StyledSubhead>{subtitle}</StyledSubhead>
      <StyledParagraph>
        {body.split("\n").map((line, i) => (
          <React.Fragment key={i}>
            {line}
            <br />
          </React.Fragment>
        ))}
      </StyledParagraph>
    </StyledSection09>
  );
};
