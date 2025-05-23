import React from "react";
import styled from "styled-components";

const SectionWrapper = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  position: relative;
`;

const Heading = styled.div`
  position: absolute;
  left: 53px;
  top: 136px;
  width: 948px;
  color: black;
  font-size: 50px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

const Subheading = styled.div`
  position: absolute;
  left: 53px;
  top: 188px;
  width: 948px;
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

const BodyText = styled.div`
  position: absolute;
  left: 20px;
  top: 548px;
  width: 1014px;
  text-align: right;
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
`;

export const Section13 = ({ title, subtitle, body }) => {
  return (
    <SectionWrapper>
      <Heading>{title}</Heading>
      <Subheading>{subtitle}</Subheading>
      <BodyText>
        {body.split("\n").map((line, i) => (
          <React.Fragment key={i}>
            {line}
            <br />
          </React.Fragment>
        ))}
      </BodyText>
    </SectionWrapper>
  );
};
